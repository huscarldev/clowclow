"""Unit tests for DynamicModelBuilder - testing schema handling and model creation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from clowclow.dynamic_model_builder import DynamicModelBuilder


class TestSchemaTypeConversion:
    """Test JSON schema to Python type conversion."""

    @pytest.mark.parametrize("schema,expected_type", [
        ({"type": "string"}, str),
        ({"type": "integer"}, int),
        ({"type": "number"}, float),
        ({"type": "boolean"}, bool),
    ])
    def test_get_type_from_schema_primitives(self, schema, expected_type):
        """Test converting primitive types using parametrize."""
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == expected_type

    @pytest.mark.parametrize("items_type,expected_type", [
        ({"type": "string"}, list[str]),
        ({"type": "integer"}, list[int]),
        ({"type": "number"}, list[float]),
        ({"type": "boolean"}, list[bool]),
    ])
    def test_get_type_from_schema_arrays(self, items_type, expected_type):
        """Test converting array types using parametrize."""
        schema = {"type": "array", "items": items_type}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == expected_type

    @pytest.mark.parametrize("value_type,expected_type", [
        ({"type": "string"}, dict[str, str]),
        ({"type": "integer"}, dict[str, int]),
        ({"type": "number"}, dict[str, float]),
        ({"type": "boolean"}, dict[str, bool]),
    ])
    def test_get_type_from_schema_objects(self, value_type, expected_type):
        """Test converting object types using parametrize."""
        schema = {"type": "object", "additionalProperties": value_type}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == expected_type

    def test_get_type_from_schema_anyof_with_null(self):
        """Test converting anyOf with null (optional field)."""
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == str | None

    def test_get_type_from_schema_anyof_without_null(self):
        """Test converting anyOf without null."""
        schema = {"anyOf": [{"type": "string"}]}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == str

    def test_get_type_from_schema_ref(self):
        """Test converting $ref reference."""
        schema = {"$ref": "#/$defs/SomeModel"}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == dict

    def test_get_type_from_schema_unknown_defaults_to_string(self):
        """Test that unknown types default to string."""
        schema = {"type": "unknown"}
        result = DynamicModelBuilder.get_type_from_schema(schema)
        assert result == str


class TestSchemaRefResolution:
    """Test JSON schema reference resolution."""

    def test_resolve_simple_ref(self):
        """Test resolving a simple $ref reference."""
        schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        resolved = DynamicModelBuilder.resolve_schema_refs(schema)

        # The $ref should be inlined
        assert resolved["properties"]["user"]["type"] == "object"
        assert resolved["properties"]["user"]["properties"]["name"]["type"] == "string"

    def test_resolve_nested_refs(self):
        """Test resolving nested $ref references."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"$ref": "#/$defs/Data"}
            },
            "$defs": {
                "Data": {
                    "type": "object",
                    "properties": {
                        "user": {"$ref": "#/$defs/User"}
                    }
                },
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        resolved = DynamicModelBuilder.resolve_schema_refs(schema)

        # Both refs should be inlined
        assert resolved["properties"]["data"]["type"] == "object"
        assert resolved["properties"]["data"]["properties"]["user"]["type"] == "object"
        assert resolved["properties"]["data"]["properties"]["user"]["properties"]["name"]["type"] == "string"

    def test_resolve_array_of_refs(self):
        """Test resolving $ref in array items."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/User"}
                }
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        resolved = DynamicModelBuilder.resolve_schema_refs(schema)

        assert resolved["properties"]["users"]["items"]["type"] == "object"
        assert resolved["properties"]["users"]["items"]["properties"]["name"]["type"] == "string"


class TestModelCreation:
    """Test dynamic Pydantic model creation."""

    def test_create_simple_model(self):
        """Test creating a simple model from schema."""
        schema = {
            "type": "object",
            "title": "TestModel",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

        Model = DynamicModelBuilder.create_model_from_schema(schema)

        # Test model creation
        instance = Model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_create_model_with_optional_fields(self):
        """Test creating model with optional fields."""
        schema = {
            "type": "object",
            "title": "TestModel",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"}
            },
            "required": ["required_field"]
        }

        Model = DynamicModelBuilder.create_model_from_schema(schema)

        # Should work with only required field
        instance = Model(required_field="value")
        assert instance.required_field == "value"
        assert instance.optional_field is None

    def test_create_model_with_arrays(self):
        """Test creating model with array fields."""
        schema = {
            "type": "object",
            "title": "TestModel",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["tags"]
        }

        Model = DynamicModelBuilder.create_model_from_schema(schema)

        instance = Model(tags=["tag1", "tag2"])
        assert instance.tags == ["tag1", "tag2"]

    def test_create_model_with_nested_objects(self):
        """Test creating model with nested object fields."""
        schema = {
            "type": "object",
            "title": "TestModel",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }
            },
            "required": []
        }

        Model = DynamicModelBuilder.create_model_from_schema(schema)

        instance = Model(user={"name": "Alice", "email": "alice@example.com"})
        assert instance.user["name"] == "Alice"
        assert instance.user["email"] == "alice@example.com"


class TestPostProcessing:
    """Test model data post-processing."""

    def test_post_process_none_to_empty_list(self):
        """Test that None array values are converted to []."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        class TestModel(BaseModel):
            items: list[str] | None = None

        instance = TestModel()
        processed = DynamicModelBuilder.post_process_model_data(instance.model_dump(), schema)

        assert processed["items"] == []

    def test_post_process_none_to_empty_dict(self):
        """Test that None object values are converted to {}."""
        schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }

        class TestModel(BaseModel):
            metadata: dict[str, str] | None = None

        instance = TestModel()
        processed = DynamicModelBuilder.post_process_model_data(instance.model_dump(), schema)

        assert processed["metadata"] == {}

    def test_post_process_preserves_actual_values(self):
        """Test that actual values are preserved during post-processing."""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object", "additionalProperties": {"type": "string"}}
            }
        }

        class TestModel(BaseModel):
            items: list[str] = ["item1"]
            metadata: dict[str, str] = {"key": "value"}

        instance = TestModel()
        processed = DynamicModelBuilder.post_process_model_data(instance.model_dump(), schema)

        assert processed["items"] == ["item1"]
        assert processed["metadata"] == {"key": "value"}

    def test_post_process_nested_none_values(self):
        """Test that nested None values are NOT automatically converted (only top-level)."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }

        data = {
            "data": {
                "items": None
            }
        }

        processed = DynamicModelBuilder.post_process_model_data(data, schema)

        # Post-processing only handles top-level None values, not nested ones
        assert processed["data"]["items"] is None
