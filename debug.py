from pydantic import BaseModel
import json
class Address(BaseModel):
            street: str
            city: str
            country: str

class Person(BaseModel):
            name: str
            address: Address

def _resolve_schema_refs(schema: dict) -> dict:
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

schema = Person.model_json_schema()
resolved_schema = _resolve_schema_refs(schema)
schema_json = json.dumps(resolved_schema, indent=2)

print( schema_json)