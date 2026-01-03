"""Serialization helpers for profiling schemas."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from types import UnionType
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

SCHEMA_VERSION = "1.0"

T = TypeVar("T", bound="SerializableMixin")


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and is_dataclass(tp)


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return non_none[0] if non_none else tp
    if origin is Optional:
        return args[0] if args else tp
    if origin is None:
        return tp
    if origin is list:
        return list
    return tp


def _serialize_value(value: Any) -> Any:
    if isinstance(value, SerializableMixin):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _construct_value(value: Any, tp: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(tp)
    args = get_args(tp)

    if origin in (Optional, Union, UnionType):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        target = non_none[0] if non_none else Any
        return _construct_value(value, target)

    if origin is list and args:
        return [_construct_value(v, args[0]) for v in value]

    if origin is dict and len(args) == 2:
        key_type, val_type = args
        return {
            _construct_value(k, key_type): _construct_value(v, val_type)
            for k, v in value.items()
        }

    concrete = _unwrap_optional(tp)
    if _is_dataclass_type(concrete) and hasattr(concrete, "from_dict"):
        return concrete.from_dict(value)

    return value


class SerializableMixin:
    """Mixin that adds JSON serialization helpers."""

    def to_dict(self) -> Dict[str, Any]:
        payload = {f.name: _serialize_value(getattr(self, f.name)) for f in fields(self)}
        payload["_schema_version"] = SCHEMA_VERSION
        payload["_schema_type"] = self.__class__.__name__
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        type_hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for field in fields(cls):
            field_type = type_hints.get(field.name, field.type)
            if field.name in clean:
                kwargs[field.name] = _construct_value(clean[field.name], field_type)
            elif field.default is not MISSING:
                kwargs[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[field.name] = field.default_factory()  # type: ignore[misc]
            else:
                kwargs[field.name] = None
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls: Type[T], raw: str) -> T:
        return cls.from_dict(json.loads(raw))
