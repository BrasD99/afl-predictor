from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    championships: Mapped[list[Championship]] = relationship(back_populates="league")


class Championship(Base):
    __tablename__ = "championships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    league: Mapped[League] = relationship(back_populates="championships")
    seasons: Mapped[list[Season]] = relationship(back_populates="championship")


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    championship_id: Mapped[int] = mapped_column(ForeignKey("championships.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    in_progress: Mapped[bool] = mapped_column(Boolean, default=False)
    starting_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ending_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    championship: Mapped[Championship] = relationship(back_populates="seasons")
    matches: Mapped[list[Match]] = relationship(back_populates="season")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    logo_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("external_id", name="uq_matches_external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), nullable=False, index=True)
    state_code: Mapped[int] = mapped_column(Integer, nullable=False)
    played_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    tour_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    team_home_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    team_away_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    score_ft_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_ft_away: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_pen_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_pen_away: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tech_defeat: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    stage_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stadium_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    season: Mapped[Season] = relationship(back_populates="matches")
    team_home: Mapped[Team] = relationship(foreign_keys=[team_home_id])
    team_away: Mapped[Team] = relationship(foreign_keys=[team_away_id])


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)


def get_session_factory(db_path: str):
    return sessionmaker(bind=get_engine(db_path), expire_on_commit=False)


def init_db(db_path: str) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
