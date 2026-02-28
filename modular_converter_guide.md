# Modular-to-Modeling Converter: Pitfalls & Rules

Hard-won lessons from developing `modular_qwen3.py`. Feed this document to AI tools
working on modular files so they don't repeat the same mistakes.

**Converter command:**
```bash
cd transformers
python utils/modular_model_converter.py --files_to_parse src/transformers/models/qwen3/modular_qwen3.py
# or shorthand:
python utils/modular_model_converter.py qwen3
```

---

## 1. Module-Level Assignments: Plain `Assign` Only

**The converter's `visit_SimpleStatementLine` for module-level code only matches `m.Assign(...)` nodes.
Python annotated assignments (`AnnAssign`) are silently skipped.**

```python
# ✅ GOOD — converter tracks this
_FP4_E2M1_GRID_LIST = _build_fp4_e2m1_grid()
_SOME_CONSTANT = 42

# ❌ BAD — converter silently ignores this, function/variable won't appear in modeling.py
_FP4_E2M1_GRID_LIST: list[float] = _build_fp4_e2m1_grid()
vals: set[float] = set()
```

**Rule:** Never use type annotations on module-level variable assignments. The variable and
all functions it depends on will be missing from the generated `modeling.py`.

**Class-level attributes DO support `AnnAssign`** — this restriction only applies to module-level.

---

## 2. `super().forward(**super_kwargs)` Inside Control Flow

**The converter expands `super().forward(**super_kwargs)` by inlining the parent's forward body
and replacing `**super_kwargs` with the parent's explicit parameter list. However, the variable
name `super_kwargs` is NOT available in the generated code.**

```python
# ✅ GOOD — works if super().forward() is the only statement or a direct return
def forward(self, **super_kwargs):
    return super().forward(**super_kwargs)

# ❌ BAD — converter inlines the parent body but `super_kwargs` doesn't exist as a variable
def forward(self, **super_kwargs):
    self.setup()
    try:
        return super().forward(**super_kwargs)   # ← `super_kwargs` is undefined in generated code
    finally:
        self.cleanup()
```

The generated code will have the full parent signature (`input_ids`, `attention_mask`, etc.)
but the `try: return super().forward(**super_kwargs)` line stays as-is with the undefined
`super_kwargs` reference.

**Workaround:** When you need custom logic (try/finally, pre/post hooks) around the parent forward:
1. Write the modular with `super().forward(**super_kwargs)` as usual
2. After running the converter, **manually fix** the generated `modeling.py`:
   replace `return super().forward(**super_kwargs)` with the actual inlined body inside your
   control flow structure
3. This manual fix only needs to happen once per converter run

**Alternative:** Write the full forward body explicitly in the modular file instead of using
`super().forward()`. This avoids the issue but means duplicating the parent code.

---

## 3. Class-Level Attributes: Must Be Proper Assignments

Class-level variables must be either plain `Assign` or `AnnAssign`. The converter handles both
for class attributes, but watch out:

```python
class MyModel(ParentModel):
    # ✅ Both work for class-level attributes
    _msd_context = None                        # plain Assign
    FORMAT_MAX: float = 1.0                    # AnnAssign (OK at class level)

    # ❌ The converter won't merge these into the parent's class body:
    # - Assignments inside `if` blocks
    # - Assignments inside `for` loops
    # - Comprehensions that produce class attributes
```

---

## 4. `del self.attribute` Only Removes the Assignment

`del self.attribute` after `super().__init__()` removes the `self.attribute = ...` assignment line
from the unraveled parent body. It does NOT remove other references to `self.attribute`.

```python
class Parent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip_qkv = config.clip_qkv
        if self.clip_qkv:          # ← this stays even after del
            self.do_stuff()

class Child(Parent):
    def __init__(self, config):
        super().__init__(config)
        del self.clip_qkv          # only removes the assignment, not the `if self.clip_qkv:` block
```

**Rule:** If you `del` an attribute, make sure either (a) it's only used in the assignment line,
or (b) you also override the method that references it.

---

## 5. Dependency Tracing: Implicit Class Creation

If a parent class uses `self.mlp = ParentMLP(config)` and you don't define `ChildMLP`,
the converter automatically creates `ChildMLP(ParentMLP): pass`.

This means:
- Classes you didn't write may appear in `modeling.py`
- Their names are derived from the parent prefix → your model prefix
- If you use non-standard naming (e.g., `MyModelIncredibleMLP` instead of `MyModelMLP`),
  the converter may create both `MyModelMLP` (auto) and `MyModelIncredibleMLP` (explicit)

**Rule:** Use consistent `{ModelName}{Component}` naming for all classes.

---

## 6. Functions Imported from Other Models

Functions imported from other model files are copied into the generated `modeling.py`:

```python
# In modular_qwen3.py:
from ..qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward

# → These functions (and their transitive dependencies like rotate_half)
#   are copied into modeling_qwen3.py
```

**Important:**
- Transitive dependencies (functions called by imported functions) are also copied
- If the imported function uses module-level variables from its source file, those are
  also copied, but ONLY if they're plain `Assign` (see Rule 1)
- Circular imports between modular files are not supported

---

## 7. Decorator Inheritance

If you override a method without specifying decorators, the parent's decorators are inherited.
If you specify ANY decorator, ALL parent decorators are replaced.

```python
# Parent has @add_start_docstrings and @replace_return_docstrings
class Child(Parent):
    def forward(self, ...):      # ← inherits BOTH parent decorators
        ...

    @my_decorator
    def forward(self, ...):      # ← ONLY @my_decorator, parent decorators dropped
        ...
```

---

## 8. Docstring Variables

Variables containing `DOCSTRING` in their name have special handling:
- Set to `None` → reuses parent's docstring with names replaced
- Set to a string → uses your string
- These are module-level but get special treatment unlike other `AnnAssign` variables

```python
QWEN3_INPUTS_DOCSTRING = None    # ✅ auto-inherits from parent with name substitution
```

---

## 9. The Generated File Header Warning

Every generated `modeling.py` starts with:
```
# 🚨 This file was automatically generated from modular_*.py.
#    Do NOT edit this file manually as any edits will be overwritten.
```

However, if the converter produces broken code (e.g., the `super_kwargs` issue), you must
manually fix `modeling.py` and be aware that re-running the converter will overwrite your fix.

**Recommended workflow:**
1. Edit `modular_qwen3.py`
2. Run converter
3. Check `modeling.py` for errors (`get_errors` or run the model)
4. If the converter produces broken code, fix `modeling.py` manually
5. Document what was manually fixed (this is important for future converter runs)
6. After each future converter run, re-apply the same manual fix

---

## 10. Ordering in the Generated File

The converter reorders code in the generated file:
- Imports first (from stdlib, then third-party, then local)
- Module-level constants and functions (in dependency order)
- Classes (in dependency order: base classes before derived)

This means the order in your modular file doesn't necessarily match the order in `modeling.py`.
Don't rely on specific ordering for correctness.

---

## 11. Things That Don't Survive Conversion

| Pattern | Survives? | Workaround |
|---------|-----------|------------|
| Module-level `AnnAssign` (`x: T = v`) | ❌ No | Use plain `x = v` |
| `super().forward(**super_kwargs)` in try/finally | ❌ Broken | Manual fix in modeling.py |
| Comments above module-level assignments | ⚠️ Sometimes lost | Add comments inside functions instead |
| Conditional module-level code (`if HAS_X:`) | ⚠️ May not track deps | Test after conversion |
| `__all__` with noqa comments | ✅ Yes | Works fine |
| Class with `pass` body | ✅ Yes | Copies parent exactly |
| `del self.attr` after `super().__init__()` | ✅ Partially | Only removes assignment, not usage |
| Custom helper functions used by classes | ✅ Yes | Auto-detected as dependencies |
| Dataclasses | ⚠️ Not recommended | Use plain classes with `__init__` |

---

## 12. Quick Checklist Before Running Converter

- [ ] All module-level variables use plain `=` (no type annotations)
- [ ] All builder functions called at module level are plain `Assign` targets
- [ ] `super().forward(**super_kwargs)` is not inside try/finally/if-else
- [ ] Class names follow `{ModelName}{Component}` pattern consistently
- [ ] No circular imports between modular files
- [ ] Docstring variables set to `None` if inheriting from parent
- [ ] After conversion: check `modeling.py` for undefined variable errors
- [ ] After conversion: verify class count matches expectations

---

## 13. Our Specific Manual Fixes (Qwen3)

These fixes must be re-applied after each converter run:

### Fix 1: `super_kwargs` in try/finally (Qwen3ForCausalLM.forward)

The modular uses:
```python
try:
    return super().forward(**super_kwargs)
finally:
    MSDComputeContext.deactivate()
```

The converter generates `return super().forward(**super_kwargs)` with `super_kwargs` undefined.

**Manual fix:** Replace the `try: return super().forward(**super_kwargs)` block in `modeling_qwen3.py`
with the full inlined parent body wrapped in the try/finally:

```python
try:
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ...
    )
    # ... rest of parent forward body ...
    return CausalLMOutputWithPast(...)
finally:
    MSDComputeContext.deactivate()
```
