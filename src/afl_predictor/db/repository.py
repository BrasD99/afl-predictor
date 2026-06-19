from __future__ import annotations

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from afl_predictor.api.footballista import LeagueInfo, MatchInfo, SeasonCalendar
from afl_predictor.db.models import Championship, League, Match, Season, Team


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


class MatchRepository:
    def __init__(self, session: Session):
        self.session = session

    def match_exists(self, external_id: int) -> bool:
        stmt = select(Match.id).where(Match.external_id == external_id).limit(1)
        with self.session.no_autoflush:
            return self.session.execute(stmt).scalar_one_or_none() is not None

    def upsert_league(self, league: LeagueInfo) -> League:
        entity = self.session.get(League, league.id)
        if entity is None:
            entity = League(id=league.id, name=league.name)
            self.session.add(entity)
        else:
            entity.name = league.name
        return entity

    def upsert_championship(self, league_id: int, championship_id: int, name: str) -> Championship:
        entity = self.session.get(Championship, championship_id)
        if entity is None:
            entity = Championship(id=championship_id, league_id=league_id, name=name)
            self.session.add(entity)
        else:
            entity.name = name
            entity.league_id = league_id
        return entity

    def upsert_season(
        self,
        championship_id: int,
        season_id: int,
        name: str,
        in_progress: bool,
        starting_at: str | None,
        ending_at: str | None,
    ) -> Season:
        entity = self.session.get(Season, season_id)
        if entity is None:
            entity = Season(
                id=season_id,
                championship_id=championship_id,
                name=name,
                in_progress=in_progress,
                starting_at=_parse_datetime(starting_at),
                ending_at=_parse_datetime(ending_at),
            )
            self.session.add(entity)
        else:
            entity.championship_id = championship_id
            entity.name = name
            entity.in_progress = in_progress
            entity.starting_at = _parse_datetime(starting_at)
            entity.ending_at = _parse_datetime(ending_at)
        return entity

    def upsert_team(self, team_id: int, name: str, short_name: str | None, logo_id: int | None) -> Team:
        entity = self.session.get(Team, team_id)
        if entity is None:
            entity = Team(id=team_id, name=name, short_name=short_name, logo_id=logo_id)
            self.session.add(entity)
        else:
            entity.name = name
            entity.short_name = short_name
            entity.logo_id = logo_id
        return entity

    def save_match_if_new(self, season_id: int, match: MatchInfo) -> str:
        """Возвращает 'inserted', 'skipped' или 'ignored' (нет даты матча)."""
        played_at = _parse_datetime(match.date)
        if played_at is None:
            return "ignored"

        if self.match_exists(match.id):
            return "skipped"

        self.upsert_team(
            match.team_home.id,
            match.team_home.name,
            match.team_home.short_name,
            match.team_home.logo_id,
        )
        self.upsert_team(
            match.team_away.id,
            match.team_away.name,
            match.team_away.short_name,
            match.team_away.logo_id,
        )

        entity = Match(
            external_id=match.id,
            season_id=season_id,
            state_code=match.state_code,
            played_at=played_at,
            tour_number=match.tour_number,
            team_home_id=match.team_home.id,
            team_away_id=match.team_away.id,
            score_ft_home=match.score_ft_home,
            score_ft_away=match.score_ft_away,
            score_pen_home=match.score_pen_home,
            score_pen_away=match.score_pen_away,
            tech_defeat=match.tech_defeat,
            stage_name=match.stage.name if match.stage else None,
            stadium_name=match.stadium.name if match.stadium else None,
        )
        self.session.add(entity)
        return "inserted"

    def import_season_calendar(self, calendar: SeasonCalendar, league_id: int) -> tuple[int, int, int]:
        championship_id = calendar.championship_id or 0
        if championship_id:
            self.upsert_championship(
                league_id=league_id,
                championship_id=championship_id,
                name=calendar.championship_name or f"Championship {championship_id}",
            )
            season_championship_id = championship_id
        else:
            season_championship_id = self._ensure_placeholder_championship(league_id)

        self.upsert_season(
            championship_id=season_championship_id,
            season_id=calendar.season_id,
            name=calendar.season_name,
            in_progress=calendar.in_progress,
            starting_at=calendar.starting_at,
            ending_at=calendar.ending_at,
        )

        inserted = 0
        skipped = 0
        ignored = 0
        for match in calendar.matches:
            result = self.save_match_if_new(calendar.season_id, match)
            if result == "inserted":
                inserted += 1
            elif result == "skipped":
                skipped += 1
            else:
                ignored += 1
        return inserted, skipped, ignored

    def _ensure_placeholder_championship(self, league_id: int) -> int:
        placeholder_id = -(league_id * 1000)
        self.upsert_championship(league_id=league_id, championship_id=placeholder_id, name="Unknown")
        return placeholder_id

    def get_team(self, team_id: int) -> Team | None:
        return self.session.get(Team, team_id)

    def get_season_championship_map(self) -> dict[int, int]:
        stmt = select(Season.id, Season.championship_id)
        return {int(row[0]): int(row[1]) for row in self.session.execute(stmt)}

    def list_finished_matches(
        self,
        finished_state_code: int,
        *,
        league_id: int | None = None,
        season_id: int | None = None,
        team_ids: list[int] | None = None,
    ) -> list[Match]:
        stmt = (
            select(Match)
            .options(joinedload(Match.team_home), joinedload(Match.team_away))
            .where(
                Match.state_code == finished_state_code,
                Match.score_ft_home.is_not(None),
                Match.score_ft_away.is_not(None),
            )
        )
        if league_id is not None:
            stmt = (
                stmt.join(Season, Match.season_id == Season.id)
                .join(Championship, Season.championship_id == Championship.id)
                .where(Championship.league_id == league_id)
            )
        if season_id is not None:
            stmt = stmt.where(Match.season_id == season_id)
        if team_ids:
            stmt = stmt.where(
                or_(
                    Match.team_home_id.in_(team_ids),
                    Match.team_away_id.in_(team_ids),
                )
            )
        stmt = stmt.order_by(Match.played_at)
        return list(self.session.scalars(stmt).unique().all())

    def find_latest_common_season(
        self,
        team_a_id: int,
        team_b_id: int,
        finished_state_code: int,
        league_id: int | None = None,
    ) -> int | None:
        def seasons_for(team_id: int) -> set[int]:
            stmt = (
                select(Match.season_id)
                .where(
                    Match.state_code == finished_state_code,
                    or_(Match.team_home_id == team_id, Match.team_away_id == team_id),
                )
            )
            if league_id is not None:
                stmt = (
                    stmt.join(Season, Match.season_id == Season.id)
                    .join(Championship, Season.championship_id == Championship.id)
                    .where(Championship.league_id == league_id)
                )
            stmt = stmt.distinct()
            return {int(row[0]) for row in self.session.execute(stmt)}

        common = seasons_for(team_a_id) & seasons_for(team_b_id)
        if not common:
            return None

        stmt = (
            select(Match.season_id)
            .where(Match.season_id.in_(common))
            .order_by(Match.played_at.desc())
            .limit(1)
        )
        result = self.session.execute(stmt).scalar_one_or_none()
        return int(result) if result is not None else None
