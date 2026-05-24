# cascade_fir

Исследовательский C/C++ проект по каскадной декомпозиции КИХ-фильтров, рассчитанных методом частотной выборки.

Текущее основное направление разработки: `exact-first`.
Сначала из точных нулевых отсчётов `H[k]` извлекаются структурно известные сомножители на единичной окружности, затем уже обрабатывается уменьшенный остаток.

Старые эвристические пайплайны `v2/v3/v4` и `liquid-dsp` comparison app не участвуют в сборке. Они сохранены в `legacy_unused_archive.cpp.disabled` только как исторический архив с пояснением, почему этот код не используется.

## Layout

- `src/decompose_exact_fs.cpp` — mainline exact-first семейство (`decompose_exact_fs`, `decompose_exact_fs_structured`, `decompose_exact_fs_full`).
- `src/cascade_decompose.cpp` — компактный baseline `decompose()` на последовательном выделении известных UC-нулей.
- `examples/exact_fs_diagnostics.cpp` — основной диагностический executable для exact-first.
- `tests/core/` — smoke-тесты базовых модулей.
- `tests/exact/` — тесты exact-first ветки.
- `legacy_unused_archive.cpp.disabled` — некомпилируемый архив старых экспериментальных файлов.

## Build

Минимальная сборка exact-first:

```bash
cmake -S . -B build
cmake --build build -j
```

Запуск основных тестов:

```bash
ctest --test-dir build --output-on-failure -L core
ctest --test-dir build --output-on-failure -L exact_fs
```

Запуск диагностики exact-first:

```bash
./build/exact_fs_diagnostics
```

## Dependencies

- FFTW3 — базовая зависимость проекта.
- `quadmath` и `lapack` — нужны для exact-first ветки.
