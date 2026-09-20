"""Validated data models for Moltbook responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Author(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: str = "unknown"


class Post(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str = "Untitled"
    content: str = ""
    author: Author = Field(default_factory=Author)
    url: str | None = None
    created_at: str | None = None
    submolt: str | None = None


class Comment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    content: str = ""
    author: Author = Field(default_factory=Author)
    parent_id: str | None = None
    post_id: str | None = None
    created_at: str | None = None
    replies: list["Comment"] = Field(default_factory=list)


Comment.model_rebuild()
