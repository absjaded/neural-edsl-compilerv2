"""Typed scene language and its deterministic interpreter.

The learned front end supplies fields. This module validates, renders, parses
and executes those fields, so accepting an expression has a precise meaning.
"""
from dataclasses import dataclass
import re

COLORS = ("red","green","blue")
SHAPES = ("circle","square","triangle")
POSITIONS = ("left","right")
ROLES = ("a","b")


@dataclass(frozen=True)
class Object:
    role: str
    color: str
    shape: str
    position: str

    def __post_init__(self):
        for field,allowed in (("role",ROLES),("color",COLORS),("shape",SHAPES),("position",POSITIONS)):
            if getattr(self,field) not in allowed:
                raise ValueError(f"Invalid {field}: {getattr(self,field)!r}")


@dataclass(frozen=True)
class Scene:
    objects: tuple[Object,...]

    def __post_init__(self):
        if len(self.objects) != 2 or {obj.role for obj in self.objects} != set(ROLES):
            raise ValueError("A scene requires exactly one a and one b")

    def by_role(self,role):
        if role not in ROLES:
            raise ValueError("Unknown role")
        return next(obj for obj in self.objects if obj.role == role)


def render(scene):
    parts = [f"(object {obj.role} {obj.color} {obj.shape} {obj.position})"
             for obj in sorted(scene.objects,key=lambda obj:obj.role)]
    return "(scene "+" ".join(parts)+")"


def parse(text):
    tokens = re.findall(r"\(|\)|[^\s()]+",text)
    position = 0
    def take(expected=None):
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Unexpected end of program")
        token = tokens[position]
        position += 1
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        return token
    take("(")
    take("scene")
    objects = []
    for _ in ROLES:
        take("(")
        take("object")
        objects.append(Object(take(),take(),take(),take()))
        take(")")
    take(")")
    if position != len(tokens):
        raise ValueError("Trailing tokens")
    return Scene(tuple(objects))


def compile_fields(fields):
    if len(fields) != 2 or any(len(row) != 3 for row in fields):
        raise ValueError("Expected two rows of color, shape and position indices")
    objects = []
    for role,row in zip(ROLES,fields):
        for index,options in zip(row,(COLORS,SHAPES,POSITIONS)):
            if not isinstance(index,int) or isinstance(index,bool) or not 0 <= index < len(options):
                raise ValueError("Field index out of range")
        objects.append(Object(role,COLORS[row[0]],SHAPES[row[1]],POSITIONS[row[2]]))
    return Scene(tuple(objects))


def fields(scene):
    return [[COLORS.index(scene.by_role(role).color),SHAPES.index(scene.by_role(role).shape),
             POSITIONS.index(scene.by_role(role).position)] for role in ROLES]


def execute(scene,operation,*arguments):
    if operation == "select":
        if len(arguments) != 2:
            raise ValueError("select expects field and value")
        field,value = arguments
        allowed = {"color":COLORS,"shape":SHAPES,"position":POSITIONS}
        if field not in allowed or value not in allowed[field]:
            raise ValueError("Invalid selection")
        return tuple(role for role in ROLES if getattr(scene.by_role(role),field) == value)
    if operation == "count":
        return len(execute(scene,"select",*arguments))
    if operation == "left_of":
        if len(arguments) != 2:
            raise ValueError("left_of expects two roles")
        first,second = (scene.by_role(role) for role in arguments)
        return POSITIONS.index(first.position) < POSITIONS.index(second.position)
    raise ValueError("Unknown operation")


def query_signature(scene):
    """A deterministic suite that checks every property and both directions."""
    answers = []
    for field,values in (("color",COLORS),("shape",SHAPES),("position",POSITIONS)):
        for value in values:
            answers.append(execute(scene,"select",field,value))
            answers.append(execute(scene,"count",field,value))
    answers.extend([execute(scene,"left_of","a","b"),execute(scene,"left_of","b","a")])
    return tuple(answers)
