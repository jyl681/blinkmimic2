# Third-party notices

Portions of `src/blinkmimic2/` and `tools/` come from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Chong Zhang) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), by way of
[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen), used under their MIT Licenses, reproduced below. The
BONES-SEED keyword filter in `tools/prepare_bones.py` follows
[SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep) (MIT).

## BlinkMimic

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, NumPy, Pillow) are installed from their own
distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.

Training data includes Motion Data by Bones Studio ([https://bones.studio/](https://bones.studio/)). Use of the
underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. The data are not included here;
`tools/prepare_bones.py` converts a copy that you download yourself after accepting that license. Files derived from
the dataset (rendered animations, trained policies) are not covered by this repository's MIT License.
