"""Dynamic Pydantic model builder from JSON schemas."""

from __future__ import annotations

from typing import Type, TypeVar, Any
from pydantic import BaseModel, create_model


T = TypeVar('T', bound=BaseModel)


class DynamicModelBuilder:
    """Builds dynamic Pydantic models from JSON schemas."""

    @staticmethod
    def resolve_schema_refs(schema: dict) -> dict:
        """Resolve $ref references in a JSON schema by inlining definitions.

        Converts schemas with $defs and $ref into flat schemas with nested objects inline.

        Args:
            schema: JSON schema with potential $ref references

        Returns:
            Schema with all $ref references resolved and inlined
        """
        def resolve_ref(ref_path: str, root_schema: dict) -> dict:
            """Resolve a $ref path like '#/$defs/Address' to its definition."""
            if not ref_path.startswith('#/'):
                raise ValueError(f"Only local refs supported, got: {ref_path}")

            parts = ref_path[2:].split('/')  # Remove '#/' and split
            current = root_schema
            for part in parts:
                current = current[part]
            return current

        def resolve_object(obj: dict, root_schema: dict) -> dict:
            """Recursively resolve all $ref in an object."""
            if isinstance(obj, dict):
                if '$ref' in obj:
                    # Replace the $ref with the actual definition
                    ref_def = resolve_ref(obj['$ref'], root_schema)
                    # Recursively resolve the referenced definition
                    return resolve_object(ref_def, root_schema)
                else:
                    # Recursively resolve nested objects
                    return {k: resolve_object(v, root_schema) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_object(item, root_schema) for item in obj]
            else:
                return obj

        # Create a copy and resolve all refs
        resolved = resolve_object(schema, schema)

        # Remove $defs from the final schema since everything is inlined
        if isinstance(resolved, dict) and '$defs' in resolved:
            resolved = {k: v for k, v in resolved.items() if k != '$defs'}

        return resolved

    @staticmethod
    def get_type_from_schema(field_schema: dict) -> type:
        """Convert JSON schema type to Python type.

        Args:
            field_schema: JSON schema for a field

        Returns:
            Python type annotation
        """
        # Handle $ref references (nested models)
        if '$ref' in field_schema:
            # For $ref, treat as dict since we'll handle nested validation separately
            return dict

        # Handle anyOf (union types, including optional fields with None)
        if 'anyOf' in field_schema:
            any_of = field_schema['anyOf']
            # Check if this is an optional field (union with null)
            has_null = any(item.get('type') == 'null' for item in any_of)
            # Get the non-null type
            non_null_types = [item for item in any_of if item.get('type') != 'null']
            if non_null_types:
                base_type = DynamicModelBuilder.get_type_from_schema(non_null_types[0])
                if has_null:
                    return base_type | None
                return base_type
            return str  # Fallback

        schema_type = field_schema.get('type', 'string')

        if schema_type == 'string':
            return str
        elif schema_type == 'integer':
            return int
        elif schema_type == 'number':
            return float
        elif schema_type == 'boolean':
            return bool
        elif schema_type == 'array':
            # Handle arrays like list[str], list[int], etc.
            items_schema = field_schema.get('items', {})
            item_type = DynamicModelBuilder.get_type_from_schema(items_schema)
            return list[item_type]
        elif schema_type == 'object':
            # Handle objects like dict[str, int], dict[str, str], etc.
            additional_properties = field_schema.get('additionalProperties', {})
            if additional_properties:
                value_type = DynamicModelBuilder.get_type_from_schema(additional_properties)
                return dict[str, value_type]
            else:
                # If no additionalProperties, treat as generic dict
                return dict[str, str]
        else:
            # Default fallback
            return str

    @staticmethod
    def create_model_from_schema(schema_dict: dict) -> Type[BaseModel]:
        """Create a dynamic Pydantic model from a JSON schema.

        Args:
            schema_dict: JSON schema dictionary

        Returns:
            Dynamically created Pydantic model class
        """
        fields = {}
        properties = schema_dict.get('properties', {})
        required = schema_dict.get('required', [])

        for field_name, field_schema in properties.items():
            field_type = DynamicModelBuilder.get_type_from_schema(field_schema)

            # Make field optional if not required
            if field_name not in required:
                # Check if the schema specifies a default value
                if 'default' in field_schema:
                    # Use the default value from the schema
                    fields[field_name] = (field_type, field_schema['default'])
                # Provide default values for optional fields without explicit defaults
                elif field_schema.get('type') == 'array':
                    fields[field_name] = (field_type, [])
                elif field_schema.get('type') == 'object':
                    fields[field_name] = (field_type, {})
                else:
                    # Only add | None if the type doesn't already include None
                    # (anyOf with null already returns type | None)
                    if not ('anyOf' in field_schema):
                        field_type = field_type | None
                    fields[field_name] = (field_type, None)
            else:
                # Required field - check if it has a default (shouldn't normally, but handle it)
                if 'default' in field_schema:
                    fields[field_name] = (field_type, field_schema['default'])
                else:
                    fields[field_name] = (field_type, ...)

        # Create the dynamic model
        model_name = schema_dict.get('title', 'OutputModel')
        return create_model(model_name, **fields)

    @staticmethod
    def post_process_model_data(model_data: dict, schema_dict: dict) -> dict:
        """Post-process model data to handle None values for arrays and objects.

        Args:
            model_data: Model data dictionary
            schema_dict: JSON schema dictionary

        Returns:
            Post-processed model data
        """
        properties = schema_dict.get('properties', {})
        result = model_data.copy()

        for field_name, field_schema in properties.items():
            if field_schema.get('type') == 'array' and result.get(field_name) is None:
                result[field_name] = []
            elif field_schema.get('type') == 'object' and result.get(field_name) is None:
                result[field_name] = {}

        return result
