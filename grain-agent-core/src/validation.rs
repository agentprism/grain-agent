//! Tool-argument schema validation and coercion.
//!
//! Ports `validateToolArguments` and the JSON-schema coercion helpers from
//! upstream `packages/ai/src/utils/validation.ts` (pi @ 34239180).
//!
//! Upstream distinguishes TypeBox schemas (validated via typebox `Compile`,
//! coerced via `Value.Convert`) from plain JSON-Schema objects, which
//! additionally run `coerceWithJsonSchema` (validation.ts:283-295). The Rust
//! `ToolDefinition.parameters` is always a plain JSON value, so this module
//! ports the plain-JSON-Schema path: `coerceWithJsonSchema` (whose
//! `coercePrimitiveByType` subsumes the conversions `Value.Convert` performs
//! for these schemas) followed by a full JSON-Schema check via the
//! `jsonschema` crate (the counterpart of typebox `Compile(...).Check`).

use serde_json::{Map, Value};

/// Port of `getSchemaTypes` (validation.ts:19-27).
fn get_schema_types(schema: &Value) -> Vec<String> {
    match schema.get("type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|t| t.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

/// Port of `matchesJsonType` (validation.ts:29-48).
fn matches_json_type(value: &Value, ty: &str) -> bool {
    match ty {
        "number" => value.is_number(),
        // JS `Number.isInteger` accepts integral floats (e.g. 2.0).
        "integer" => value
            .as_f64()
            .map(|n| n.fract() == 0.0 && n.is_finite())
            .unwrap_or(false),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn f64_to_number(n: f64) -> Value {
    // Mirror JS number semantics: JSON output for an integral double is the
    // integer form (JS has a single number type).
    if n.fract() == 0.0 && n.is_finite() && n.abs() <= i64::MAX as f64 {
        Value::from(n as i64)
    } else {
        serde_json::Number::from_f64(n)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

/// Port of `coercePrimitiveByType` (validation.ts:58-130). Returns the
/// coerced value, or a clone of the original when no coercion applies.
fn coerce_primitive_by_type(value: &Value, ty: &str) -> Value {
    match ty {
        "number" => {
            if value.is_null() {
                return Value::from(0);
            }
            if let Some(s) = value.as_str()
                && !s.trim().is_empty()
                && let Ok(parsed) = s.trim().parse::<f64>()
                && parsed.is_finite()
            {
                return f64_to_number(parsed);
            }
            if let Some(b) = value.as_bool() {
                return Value::from(if b { 1 } else { 0 });
            }
            value.clone()
        }
        "integer" => {
            if value.is_null() {
                return Value::from(0);
            }
            if let Some(s) = value.as_str()
                && !s.trim().is_empty()
                && let Ok(parsed) = s.trim().parse::<f64>()
                // JS `Number.isInteger(parsed)`
                && parsed.fract() == 0.0
                && parsed.is_finite()
            {
                return Value::from(parsed as i64);
            }
            if let Some(b) = value.as_bool() {
                return Value::from(if b { 1 } else { 0 });
            }
            value.clone()
        }
        "boolean" => {
            if value.is_null() {
                return Value::Bool(false);
            }
            if let Some(s) = value.as_str() {
                if s == "true" {
                    return Value::Bool(true);
                }
                if s == "false" {
                    return Value::Bool(false);
                }
            }
            if let Some(n) = value.as_f64() {
                // JS `value === 1` / `value === 0`.
                if n == 1.0 {
                    return Value::Bool(true);
                }
                if n == 0.0 {
                    return Value::Bool(false);
                }
            }
            value.clone()
        }
        "string" => {
            if value.is_null() {
                return Value::String(String::new());
            }
            if let Some(n) = value.as_i64() {
                return Value::String(n.to_string());
            }
            if let Some(n) = value.as_u64() {
                return Value::String(n.to_string());
            }
            if let Some(n) = value.as_f64() {
                // JS `String(1.5)` == "1.5"; ryu-style shortest form is close
                // enough for the f64 case.
                return Value::String(format_js_number(n));
            }
            if let Some(b) = value.as_bool() {
                return Value::String(b.to_string());
            }
            value.clone()
        }
        "null" => {
            // JS `value === "" || value === 0 || value === false`.
            let is_zero = value.as_f64() == Some(0.0);
            if value.as_str() == Some("") || is_zero || value == &Value::Bool(false) {
                return Value::Null;
            }
            value.clone()
        }
        _ => value.clone(),
    }
}

/// JS `String(number)`-style formatting: integral doubles print without a
/// fractional part.
fn format_js_number(n: f64) -> String {
    if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e21 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// Port of `applySchemaObjectCoercion` (validation.ts:132-153).
fn apply_schema_object_coercion(value: &mut Map<String, Value>, schema: &Value) {
    let properties = schema.get("properties").and_then(|p| p.as_object());

    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            if let Some(v) = value.get(key) {
                let coerced = coerce_with_json_schema(v.clone(), property_schema);
                value.insert(key.clone(), coerced);
            }
        }
    }

    if let Some(additional) = schema.get("additionalProperties")
        && additional.is_object()
    {
        let defined: std::collections::HashSet<&String> =
            properties.map(|p| p.keys().collect()).unwrap_or_default();
        let keys: Vec<String> = value
            .keys()
            .filter(|k| !defined.contains(k))
            .cloned()
            .collect();
        for key in keys {
            let coerced = coerce_with_json_schema(value[&key].clone(), additional);
            value.insert(key, coerced);
        }
    }
}

/// Port of `applySchemaArrayCoercion` (validation.ts:155-172).
fn apply_schema_array_coercion(value: &mut [Value], schema: &Value) {
    match schema.get("items") {
        Some(Value::Array(item_schemas)) => {
            for (index, slot) in value.iter_mut().enumerate() {
                if let Some(item_schema) = item_schemas.get(index) {
                    *slot = coerce_with_json_schema(slot.clone(), item_schema);
                }
            }
        }
        Some(items) if items.is_object() => {
            for slot in value.iter_mut() {
                *slot = coerce_with_json_schema(slot.clone(), items);
            }
        }
        _ => {}
    }
}

/// Port of `coerceWithUnionSchema` (validation.ts:174-184): try each union
/// member, keeping the first coercion that validates against that member.
fn coerce_with_union_schema(value: Value, schemas: &[Value]) -> Value {
    for schema in schemas {
        let candidate = coerce_with_json_schema(value.clone(), schema);
        // `getSubSchemaValidator` returns undefined for uncompilable schemas,
        // which skips the member (validation.ts:50-56).
        if let Ok(validator) = jsonschema::validator_for(schema)
            && validator.is_valid(&candidate)
        {
            return candidate;
        }
    }
    value
}

/// Port of `coerceWithJsonSchema` (validation.ts:186-230).
fn coerce_with_json_schema(value: Value, schema: &Value) -> Value {
    let mut next_value = value;

    if let Some(all_of) = schema.get("allOf").and_then(|v| v.as_array()) {
        for nested in all_of {
            next_value = coerce_with_json_schema(next_value, nested);
        }
    }

    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        next_value = coerce_with_union_schema(next_value, any_of);
    }

    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        next_value = coerce_with_union_schema(next_value, one_of);
    }

    let schema_types = get_schema_types(schema);
    let matches_union_member = schema_types.len() > 1
        && schema_types
            .iter()
            .any(|ty| matches_json_type(&next_value, ty));
    if !schema_types.is_empty() && !matches_union_member {
        for ty in &schema_types {
            let candidate = coerce_primitive_by_type(&next_value, ty);
            if candidate != next_value {
                next_value = candidate;
                break;
            }
        }
    }

    if schema_types.iter().any(|t| t == "object")
        && let Some(map) = next_value.as_object_mut()
    {
        apply_schema_object_coercion(map, schema);
    }

    if schema_types.iter().any(|t| t == "array")
        && let Some(arr) = next_value.as_array_mut()
    {
        apply_schema_array_coercion(arr, schema);
    }

    next_value
}

/// Port of `formatValidationPath` (validation.ts:243-254): dotted instance
/// path, `root` for the schema root, and `path.property` for missing
/// required properties.
fn format_validation_path(error: &jsonschema::ValidationError<'_>) -> String {
    let base = error
        .instance_path()
        .to_string()
        .trim_start_matches('/')
        .replace('/', ".");
    if let jsonschema::error::ValidationErrorKind::Required { property } = error.kind() {
        let property = property
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| property.to_string());
        return if base.is_empty() {
            property
        } else {
            format!("{base}.{property}")
        };
    }
    if base.is_empty() { "root".into() } else { base }
}

/// Port of `validateToolArguments` (validation.ts:278-310): coerce the raw
/// tool-call arguments against the tool's JSON schema, then validate. On
/// success, returns the (potentially coerced) arguments that must be passed
/// to hooks and `execute`; on failure, returns the upstream-formatted error
/// message.
///
/// A `Null` schema means the tool declared no parameters schema (the Rust
/// `ToolDefinition.parameters` default); upstream tools always carry a
/// TypeBox schema, so this case skips validation and passes arguments
/// through unchanged.
pub fn validate_tool_arguments(
    tool_name: &str,
    parameters: &Value,
    args: Value,
) -> Result<Value, String> {
    if parameters.is_null() {
        return Ok(args);
    }

    // typebox `Compile` throws on an invalid schema; upstream surfaces that
    // through prepareToolCall's catch as an immediate error tool result
    // (validation.ts:282, agent-loop.ts:657-663).
    let validator = jsonschema::validator_for(parameters).map_err(|e| e.to_string())?;

    let original = args.clone();
    let coerced = coerce_with_json_schema(args.clone(), parameters);
    let candidate = if coerced != args {
        if args.is_object() && coerced.is_object() {
            // validation.ts:287-290 — merge the coercion into `args`.
            coerced
        } else {
            // validation.ts:292 — non-object roots return without the final
            // check when the coerced value validates, else fall back to the
            // original arguments as-is (mirrored exactly, including the
            // skipped re-validation of the fallback).
            //
            // KNOWN DIVERGENCE (array-rooted schemas only): upstream's
            // branch condition is `typeof args === "object"`, and in JS
            // `typeof [] === "object"`, so an array root takes the merge
            // branch above — the coercion is merged into `args` by
            // reference and an invalid result reaches the final Check and
            // throws. Here `Value::is_object()` is `false` for arrays, so
            // an array root whose coercion changed but is still invalid
            // takes this early return and passes the ORIGINAL args through
            // unvalidated instead of erroring. This path is unreachable via
            // LLM tool calls: `toolCall.arguments` is object-rooted in both
            // implementations (`Record<string, any>` upstream, and every
            // provider serializes tool arguments as a JSON object). Do not
            // "fix" this by adding `is_array()` to the merge branch without
            // also porting upstream's delete-keys/Object.assign in-place
            // merge semantics and a covering vector.
            return Ok(if validator.is_valid(&coerced) {
                coerced
            } else {
                args
            });
        }
    } else {
        args
    };

    if validator.is_valid(&candidate) {
        return Ok(candidate);
    }

    let errors: Vec<String> = validator
        .iter_errors(&candidate)
        .map(|error| format!("  - {}: {}", format_validation_path(&error), error))
        .collect();
    let errors = if errors.is_empty() {
        "Unknown validation error".to_string()
    } else {
        errors.join("\n")
    };

    Err(format!(
        "Validation failed for tool \"{tool_name}\":\n{errors}\n\nReceived arguments:\n{}",
        serde_json::to_string_pretty(&original).unwrap_or_else(|_| original.to_string())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn value_schema() -> Value {
        json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        })
    }

    #[test]
    fn passes_valid_arguments_through() {
        let args = json!({ "value": "hello" });
        let out = validate_tool_arguments("echo", &value_schema(), args.clone()).unwrap();
        assert_eq!(out, args);
    }

    #[test]
    fn coerces_number_to_string() {
        let out = validate_tool_arguments("echo", &value_schema(), json!({ "value": 42 })).unwrap();
        assert_eq!(out, json!({ "value": "42" }));
    }

    #[test]
    fn coerces_string_to_number_boolean_and_null() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": { "type": "number" },
                "size": { "type": "integer" },
                "flag": { "type": "boolean" },
                "note": { "type": "string" },
                "nothing": { "type": "null" }
            },
            "required": ["count", "size", "flag", "note", "nothing"]
        });
        let out = validate_tool_arguments(
            "t",
            &schema,
            json!({ "count": "1.5", "size": "42", "flag": "true", "note": 7, "nothing": 0 }),
        )
        .unwrap();
        assert_eq!(
            out,
            json!({ "count": 1.5, "size": 42, "flag": true, "note": "7", "nothing": null })
        );
    }

    #[test]
    fn coerces_null_to_type_defaults() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": { "type": "number" },
                "flag": { "type": "boolean" },
                "note": { "type": "string" }
            }
        });
        let out = validate_tool_arguments(
            "t",
            &schema,
            json!({ "count": null, "flag": null, "note": null }),
        )
        .unwrap();
        assert_eq!(out, json!({ "count": 0, "flag": false, "note": "" }));
    }

    #[test]
    fn missing_required_property_fails_with_upstream_message_shape() {
        let err = validate_tool_arguments("echo", &value_schema(), json!({})).unwrap_err();
        assert!(
            err.starts_with("Validation failed for tool \"echo\":\n"),
            "unexpected message: {err}"
        );
        assert!(err.contains("value"), "missing property path: {err}");
        assert!(err.contains("Received arguments:"), "missing args: {err}");
    }

    #[test]
    fn uncoercible_type_fails() {
        let err = validate_tool_arguments(
            "echo",
            &value_schema(),
            json!({ "value": { "nested": true } }),
        )
        .unwrap_err();
        assert!(err.starts_with("Validation failed for tool \"echo\":\n"));
    }

    #[test]
    fn union_type_keeps_matching_member_uncoerced() {
        // validation.ts:204-206 — a value matching one union member is left
        // alone.
        let schema = json!({
            "type": "object",
            "properties": { "id": { "type": ["string", "number"] } }
        });
        let out = validate_tool_arguments("t", &schema, json!({ "id": 42 })).unwrap();
        assert_eq!(out, json!({ "id": 42 }));
    }

    #[test]
    fn any_of_coerces_to_first_validating_member() {
        let schema = json!({
            "type": "object",
            "properties": {
                "when": { "anyOf": [{ "type": "number" }, { "type": "boolean" }] }
            }
        });
        let out = validate_tool_arguments("t", &schema, json!({ "when": "42" })).unwrap();
        assert_eq!(out, json!({ "when": 42 }));
    }

    #[test]
    fn nested_array_items_are_coerced() {
        let schema = json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": { "line": { "type": "integer" } }
                    }
                }
            }
        });
        let out = validate_tool_arguments(
            "t",
            &schema,
            json!({ "edits": [{ "line": "3" }, { "line": "4" }] }),
        )
        .unwrap();
        assert_eq!(out, json!({ "edits": [{ "line": 3 }, { "line": 4 }] }));
    }

    #[test]
    fn null_schema_skips_validation() {
        let out = validate_tool_arguments("t", &Value::Null, json!({ "anything": 1 })).unwrap();
        assert_eq!(out, json!({ "anything": 1 }));
    }
}
