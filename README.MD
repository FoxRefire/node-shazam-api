## Shazam Api for node-js (Fork for used in Librezam)

This library is heavily based on [Numenorean's work](https://github.com/Numenorean/ShazamAPI).

### Changes in this fork:

* Webpack build target
* Restore to 0.2.0 because 0.2.1-0 obviously decreases recognition accuracy

### Usage:

```js
import { Shazam, s16LEToSamplesArray } from 'shazam-api';
import fs from 'fs';

const shazam = new Shazam();
const fileContents = fs.readFileSync("test.pcm");
const samples = s16LEToSamplesArray(fileContents);

const songData = await shazam.recognizeSong(samples);
console.log(songData);
```

The data passed to s16LEToSamplesArray needs to be raw PCM with:
- Sample rate set to 16000Hz
- Bit depth set to 16 bits
- 1 channel of audio

Such a file can be generated with ffmpeg using the command:
```
ffmpeg -i <source file> -ar 16000 -ac 1 -f s16le test.pcm
```

Or within node or the browser thanks to [ffmpeg.wasm](https://www.npmjs.com/package/@ffmpeg/ffmpeg)
