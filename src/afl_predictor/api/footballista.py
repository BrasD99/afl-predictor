from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class TeamInfo:
    id: int
    name: str
    short_name: str | None
    logo_id: int | None


@dataclass(frozen=True)
class SeasonInfo:
    id: int
    name: str
    in_progress: bool
    starting_at: str | None
    ending_at: str | None


@dataclass(frozen=True)
class ChampionshipInfo:
    id: int
    name: str
    seasons: list[SeasonInfo]


@dataclass(frozen=True)
class LeagueInfo:
    id: int
    name: str
    championships: list[ChampionshipInfo]


@dataclass(frozen=True)
class CityInfo:
    id: int
    name: str


@dataclass(frozen=True)
class ClientInfo:
    id: int
    key: str
    name: str


@dataclass(frozen=True)
class LeagueListItem:
    id: int
    name: str
    sort_idx: int
    sports: str | None
    sports_subtype: str | None
    city: CityInfo | None
    client: ClientInfo | None


@dataclass(frozen=True)
class StageInfo:
    id: int
    name: str
    format: str | None


@dataclass(frozen=True)
class StadiumInfo:
    id: int
    name: str


@dataclass(frozen=True)
class MatchInfo:
    id: int
    state_code: int
    date: str
    tour_number: int | None
    score_ft_home: int | None
    score_ft_away: int | None
    score_pen_home: int | None
    score_pen_away: int | None
    tech_defeat: bool | None
    team_home: TeamInfo
    team_away: TeamInfo
    stage: StageInfo | None
    stadium: StadiumInfo | None


@dataclass(frozen=True)
class SeasonCalendar:
    season_id: int
    season_name: str
    championship_id: int | None
    championship_name: str | None
    in_progress: bool
    starting_at: str | None
    ending_at: str | None
    matches: list[MatchInfo]


class FootballistaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 30, page_size: int = 50):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size
        self._session = requests.Session()

    def get_leagues_list(self) -> list[LeagueListItem]:
        query = """query {
            League_list {
                _id
                name
                sortIdx
                sports
                sportsSubtype
                city { _id, name, flag, sortIdx }
                client { _id, key, name }
            }
        }"""
        payload = self._get("graphqlleagues-list", query)
        items: list[LeagueListItem] = []
        for raw in payload.get("League_list") or []:
            city_raw = raw.get("city")
            client_raw = raw.get("client")
            items.append(
                LeagueListItem(
                    id=int(raw["_id"]),
                    name=raw["name"],
                    sort_idx=int(raw.get("sortIdx") or 0),
                    sports=raw.get("sports"),
                    sports_subtype=raw.get("sportsSubtype"),
                    city=CityInfo(
                        id=int(city_raw["_id"]),
                        name=city_raw["name"],
                    )
                    if city_raw and city_raw.get("_id")
                    else None,
                    client=ClientInfo(
                        id=int(client_raw["_id"]),
                        key=client_raw["key"],
                        name=client_raw["name"],
                    )
                    if client_raw and client_raw.get("_id")
                    else None,
                )
            )
        return items

    def get_league_info(self, league_id: int) -> LeagueInfo:
        query = f"""query {{
            League(_id: {league_id}) {{
                _id
                name
                champs {{
                    _id
                    name
                    sortIdx
                    seasons {{
                        _id
                        name
                        sortIdx
                        inProgress
                        startingAt
                        endingAt
                    }}
                }}
            }}
        }}"""
        payload = self._get("graphqlleague-info", query)
        league = payload.get("League")
        if not league:
            raise ValueError(f"League {league_id} not found")

        championships: list[ChampionshipInfo] = []
        for champ in league.get("champs") or []:
            seasons = [
                SeasonInfo(
                    id=int(season["_id"]),
                    name=season["name"],
                    in_progress=bool(season.get("inProgress")),
                    starting_at=season.get("startingAt"),
                    ending_at=season.get("endingAt"),
                )
                for season in champ.get("seasons") or []
            ]
            championships.append(
                ChampionshipInfo(
                    id=int(champ["_id"]),
                    name=champ["name"],
                    seasons=seasons,
                )
            )

        return LeagueInfo(
            id=int(league["_id"]),
            name=league["name"],
            championships=championships,
        )

    def get_season_calendar(self, season_id: int) -> SeasonCalendar:
        matches: list[MatchInfo] = []
        offset = 0
        season_meta: dict[str, Any] | None = None

        while True:
            query = f"""query {{
                Season(_id: {season_id}) {{
                    _id
                    name
                    inProgress
                    startingAt
                    endingAt
                    champ {{
                        _id
                        name
                    }}
                    calendar(offset: {offset}) {{
                        total
                        items {{
                            _id
                            stateCode
                            date
                            tourNumber
                            scoreFtHome
                            scoreFtAway
                            scorePenHome
                            scorePenAway
                            techDefeat
                            stage {{
                                _id
                                name
                                format
                            }}
                            stadium {{ _id, name }}
                            teamHome {{ _id, shortName, name, logoId }}
                            teamAway {{ _id, shortName, name, logoId }}
                        }}
                    }}
                }}
            }}"""
            payload = self._get("graphqlseason-calendar", query)
            season = payload.get("Season")
            if not season:
                raise ValueError(f"Season {season_id} not found")

            if season_meta is None:
                champ = season.get("champ") or {}
                season_meta = {
                    "season_name": season["name"],
                    "in_progress": bool(season.get("inProgress")),
                    "starting_at": season.get("startingAt"),
                    "ending_at": season.get("endingAt"),
                    "championship_id": int(champ["_id"]) if champ.get("_id") else None,
                    "championship_name": champ.get("name"),
                }

            calendar = season.get("calendar") or {}
            items = calendar.get("items") or []
            if not items:
                break

            for item in items:
                matches.append(self._parse_match(item))

            if len(items) < self.page_size:
                break
            offset += len(items)

        assert season_meta is not None
        return SeasonCalendar(
            season_id=season_id,
            season_name=season_meta["season_name"],
            championship_id=season_meta["championship_id"],
            championship_name=season_meta["championship_name"],
            in_progress=season_meta["in_progress"],
            starting_at=season_meta["starting_at"],
            ending_at=season_meta["ending_at"],
            matches=matches,
        )

    def _get(self, endpoint: str, query: str) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}?query={quote(query)}"
        response = self._session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        if "errors" in body:
            raise RuntimeError(f"GraphQL errors: {body['errors']}")
        return body.get("data") or {}

    @staticmethod
    def _parse_match(item: dict[str, Any]) -> MatchInfo:
        stage_raw = item.get("stage")
        stadium_raw = item.get("stadium")
        home_raw = item["teamHome"]
        away_raw = item["teamAway"]

        return MatchInfo(
            id=int(item["_id"]),
            state_code=int(item["stateCode"]),
            date=item["date"],
            tour_number=item.get("tourNumber"),
            score_ft_home=item.get("scoreFtHome"),
            score_ft_away=item.get("scoreFtAway"),
            score_pen_home=item.get("scorePenHome"),
            score_pen_away=item.get("scorePenAway"),
            tech_defeat=item.get("techDefeat"),
            team_home=TeamInfo(
                id=int(home_raw["_id"]),
                name=home_raw["name"],
                short_name=home_raw.get("shortName"),
                logo_id=int(home_raw["logoId"]) if home_raw.get("logoId") else None,
            ),
            team_away=TeamInfo(
                id=int(away_raw["_id"]),
                name=away_raw["name"],
                short_name=away_raw.get("shortName"),
                logo_id=int(away_raw["logoId"]) if away_raw.get("logoId") else None,
            ),
            stage=StageInfo(
                id=int(stage_raw["_id"]),
                name=stage_raw["name"],
                format=stage_raw.get("format"),
            )
            if stage_raw
            else None,
            stadium=StadiumInfo(
                id=int(stadium_raw["_id"]),
                name=stadium_raw["name"],
            )
            if stadium_raw
            else None,
        )
