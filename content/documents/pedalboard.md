+++
date = '2025-10-04T14:20:22+08:00'
title = 'Pedalboard 文档'
+++

Github: https://github.com/spotify/pedalboard

Docs: https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.LowShelfFilter

# Pedalboard API Documentation

The `pedalboard` module provides classes and functions for adding effects to audio. Most classes in this module are subclasses of `Plugin`, each of which allows applying effects to an audio buffer or stream.

> **Note:** For audio I/O functionality (i.e.: reading and writing audio files), see the `pedalboard.io` module.

The `pedalboard` module is named after the concept of a guitar pedalboard, in which musicians will chain various effects pedals together to give them complete control over their sound. The `pedalboard` module implements this concept with its main `Pedalboard` class:

```python
from pedalboard import Pedalboard, Chorus, Distortion, Reverb

# Create an empty Pedalboard object:
my_pedalboard = Pedalboard()

# Treat this object like a Python list:
my_pedalboard.append(Chorus())
my_pedalboard.append(Distortion())
my_pedalboard.append(Reverb())

# Pass audio through this pedalboard:
output_audio = my_pedalboard(input_audio, input_audio_samplerate)
```

`Pedalboard` objects are lists of zero or more `Plugin` objects, and `Pedalboard` objects themselves are subclasses of `Plugin` - which allows for nesting and composition.

---

## Classes

### `pedalboard.AudioProcessorParameter`
A wrapper around various different parameters exposed by `VST3Plugin` or `AudioUnitPlugin` instances.

`AudioProcessorParameter` objects are rarely used directly, and usually used via their implicit interface:

```python
my_plugin = load_plugin("My Cool Audio Effect.vst3")
# Print all of the parameter names:
print(my_plugin.parameters.keys())
# ["mix", "delay_time_ms", "foobar"]
# Access each parameter as if it were just a Python attribute:
my_plugin.mix = 0.5
my_plugin.delay_time_ms = 400
```

> **Note:** `AudioProcessorParameter` tries to guess the range of valid parameter values, as well as the type/unit of the parameter, when instantiated. This guess may not always be accurate. Raw control over the underlying parameter’s value can be had by accessing the `raw_value` attribute, which is always bounded on [0, 1] and is passed directly to the underlying plugin object.

#### Properties
- `label: Optional[str]` – The units used by this parameter (Hz, dB, etc). May be `None` if the plugin does not expose units for this parameter or if automatic unit detection fails.
- `units: Optional[str]` – Alias for "label" – the units used by this parameter (Hz, dB, etc). May be `None` if the plugin does not expose units for this parameter or if automatic unit detection fails.

---

### `pedalboard.ExternalPlugin`
A wrapper around a third-party effect plugin.

Don’t use this directly; use one of `pedalboard.VST3Plugin` or `pedalboard.AudioUnitPlugin` instead.

#### Methods
- `__call__(*args, **kwargs)` – Overloaded function.
  - `__call__(self: pedalboard.Plugin, input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio or MIDI buffer through this plugin, returning audio. Alias for `process()`.
  - `__call__(self: pedalboard.ExternalPlugin, midi_messages: object, duration: float, sample_rate: float, num_channels: int = 2, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio or MIDI buffer through this plugin, returning audio. Alias for `process()`.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

#### Methods
- `process(*args, **kwargs)` – Overloaded function.
  - `process(self: pedalboard.ExternalPlugin, midi_messages: object, duration: float, sample_rate: float, num_channels: int = 2, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Pass a buffer of audio (as a 32- or 64-bit NumPy array) or a list of MIDI messages to this plugin, returning audio.
  - `process(self: pedalboard.Plugin, input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Pass a buffer of audio (as a 32- or 64-bit NumPy array) or a list of MIDI messages to this plugin, returning audio.

> **Note:** The `process()` method can also be used via `__call__()`; i.e.: just calling this object like a function (`my_plugin(...)`) will automatically invoke `process()` with the same arguments.

- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

---

### `pedalboard.Pedalboard(plugins: Optional[List[Plugin]] = None)`
A container for a series of `Plugin` objects, to use for processing audio, like a guitar pedalboard.

`Pedalboard` objects act like regular Python `List` objects, but come with an additional `process()` method (also aliased to `__call__()`), allowing audio to be passed through the entire `Pedalboard` object for processing:

```python
my_pedalboard = Pedalboard()
my_pedalboard.append(Reverb())
output_audio = my_pedalboard(input_audio)
```

> **Warning:** `Pedalboard` objects may only contain effects plugins (i.e.: those for which `is_effect` is `True`), and cannot contain instrument plugins (i.e.: those for which `is_instrument` is `True`).

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `append(plugin: Plugin) -> None` – Append a plugin to the end of this container.
- `insert(index: int, plugin: Plugin) -> None` – Insert a plugin at the specified index.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `remove(plugin: Plugin) -> None` – Remove a plugin by its value.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.load_plugin(path_to_plugin_file: str, parameter_values: Dict[str, Union[str, int, float, bool]] = {}, plugin_name: Optional[str] = None, initialization_timeout: float = 10.0) -> ExternalPlugin`
Load an audio plugin.

Two plugin formats are supported:
- VST3® format is supported on macOS, Windows, and Linux
- Audio Units are supported on macOS

#### Parameters
- `path_to_plugin_file (str)` – The path of a VST3® or Audio Unit plugin file or bundle.
- `parameter_values (Dict[str, Union[str, int, float, bool]])` – An optional dictionary of initial values to provide to the plugin after loading. Keys in this dictionary are expected to match the parameter names reported by the plugin, but normalized to strings that can be used as Python identifiers. (These are the same identifiers that are used as keys in the `.parameters` dictionary of a loaded plugin.)
- `plugin_name (Optional[str])` – An optional plugin name that can be used to load a specific plugin from a multi-plugin package. If a package is loaded but a `plugin_name` is not provided, an exception will be thrown.
- `initialization_timeout (float)` – The number of seconds that Pedalboard will spend trying to load this plugin. Some plugins load resources asynchronously in the background on startup; using larger values for this parameter can give these plugins time to load properly. Introduced in v0.7.6.

#### Returns
An instance of `pedalboard.VST3Plugin` or `pedalboard.AudioUnitPlugin`.

#### Throws
- `ImportError` – if the plugin cannot be found or loaded
- `RuntimeError` – if the plugin file contains more than one plugin, but no `plugin_name` was provided

---

### `pedalboard.Bitcrush`
A plugin that reduces the signal to a given bit depth, giving the audio a lo-fi, digitized sound. Floating-point bit depths are supported.

Bitcrushing changes the amount of "vertical" resolution used for an audio signal (i.e.: how many unique values could be used to represent each sample). For an effect that changes the "horizontal" resolution (i.e.: how many samples are available per second), see `pedalboard.Resample`.

#### Properties
- `bit_depth` – The bit depth to quantize the signal to. Must be between 0 and 32 bits. May be an integer, decimal, or floating-point value. Each audio sample will be quantized onto `2 ** bit_depth` values.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Chain`
Run zero or more plugins as a plugin. Useful when used with the `Mix` plugin.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `append(plugin: Plugin) -> None` – Append a plugin to the end of this container.
- `insert(index: int, plugin: Plugin) -> None` – Insert a plugin at the specified index.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `remove(plugin: Plugin) -> None` – Remove a plugin by its value.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Chorus`
A basic chorus effect.

This audio effect can be controlled via the speed and depth of the LFO controlling the frequency response, a mix control, a feedback control, and the centre delay of the modulation.

> **Note:** To get classic chorus sounds try to use a centre delay time around 7-8 ms with a low feedback volume and a low depth. This effect can also be used as a flanger with a lower centre delay time and a lot of feedback, and as a vibrato effect if the mix value is 1.

#### Properties
- `rate_hz` – The speed of the chorus effect’s low-frequency oscillator (LFO), in Hertz. This value must be between 0 Hz and 100 Hz.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Clipping`
A distortion plugin that adds hard distortion to the signal by clipping the signal at the provided threshold (in decibels).

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Compressor`
A dynamic range compressor, used to reduce the volume of loud sounds and "compress" the loudness of the signal.

For a lossy compression algorithm that introduces noise or artifacts, see `pedalboard.MP3Compressor` or `pedalboard.GSMCompressor`.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Convolution`
An audio convolution, suitable for things like speaker simulation or reverb modeling.

The convolution impulse response can be specified either by filename or as a 32-bit floating point NumPy array. If a NumPy array is provided, the `sample_rate` argument must also be provided to indicate the sample rate of the impulse response.

Support for passing NumPy arrays as impulse responses introduced in v0.9.10.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Delay`
A digital delay plugin with controllable delay time, feedback percentage, and dry/wet mix.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Distortion`
A distortion effect, which applies a non-linear (tanh, or hyperbolic tangent) waveshaping function to apply harmonically pleasing distortion to a signal.

This plugin produces a signal that is roughly equivalent to running:
```python
def distortion(x):
    return tanh(x * db_to_gain(drive_db))
```

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.ExternalPluginReloadType`
Indicates the behavior of an external plugin when `reset()` is called.

#### Members
- `Unknown` – The behavior of the plugin is unknown. This will force a full reinstantiation of the plugin every time reset is called.
- `ClearsAudioOnReset` – This plugin clears its internal buffers correctly when `reset()` is called. The plugin will not be reinstantiated when reset is called.
- `PersistsAudioOnReset` – This plugin does not clear its internal buffers as expected when `reset()` is called. This will force a full reinstantiation of the plugin every time reset is called.

#### Properties
- `name`

---

### `pedalboard.GSMFullRateCompressor`
An audio degradation/compression plugin that applies the GSM "Full Rate" compression algorithm to emulate the sound of a 2G cellular phone connection. This plugin internally resamples the input audio to a fixed sample rate of 8kHz (required by the GSM Full Rate codec), although the quality of the resampling algorithm can be specified.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Gain`
A gain plugin that increases or decreases the volume of a signal by amplifying or attenuating it by the provided value (in decibels). No distortion or other effects are applied.

Think of this as a volume control.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.HighShelfFilter`
A high shelf filter plugin with variable Q and gain, as would be used in an equalizer. Frequencies above the cutoff frequency will be boosted (or cut) by the provided gain (in decibels).

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.HighpassFilter`
Apply a first-order high-pass filter with a roll-off of 6dB/octave. The cutoff frequency will be attenuated by -3dB (i.e.: 0.707x as loud, expressed as a gain factor) and lower frequencies will be attenuated by a further 6dB per octave.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.IIRFilter`
An abstract class that implements various kinds of infinite impulse response (IIR) filter designs. This should not be used directly; use `HighShelfFilter`, `LowShelfFilter`, or `PeakFilter` directly instead.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Invert`
Flip the polarity of the signal. This effect is not audible on its own and takes no parameters. This effect is mathematically identical to `def invert(x): return -x`.

Inverting a signal may be useful to cancel out signals in many cases; for instance, `Invert` can be used with the `Mix` plugin to remove the original signal from an effects chain that contains multiple signals.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.LadderFilter`
A multi-mode audio filter based on the classic Moog synthesizer ladder filter, invented by Dr. Bob Moog in 1968.

Depending on the filter’s mode, frequencies above, below, or on both sides of the cutoff frequency will be attenuated. Higher values for the resonance parameter may cause peaks in the frequency response around the cutoff frequency.

#### Class `Mode`
The type of filter architecture to use.

##### Members
- `LPF12` – A low-pass filter with 12 dB of attenuation per octave above the cutoff frequency.
- `HPF12` – A high-pass filter with 12 dB of attenuation per octave below the cutoff frequency.
- `BPF12` – A band-pass filter with 12 dB of attenuation per octave on both sides of the cutoff frequency.
- `LPF24` – A low-pass filter with 24 dB of attenuation per octave above the cutoff frequency.
- `HPF24` – A high-pass filter with 24 dB of attenuation per octave below the cutoff frequency.
- `BPF24` – A band-pass filter with 24 dB of attenuation per octave on both sides of the cutoff frequency.

##### Properties
- `name`

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Limiter`
A simple limiter with standard threshold and release time controls, featuring two compressors and a hard clipper at 0 dB.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.LowShelfFilter`
A low shelf filter with variable Q and gain, as would be used in an equalizer. Frequencies below the cutoff frequency will be boosted (or cut) by the provided gain value.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.LowpassFilter`
Apply a first-order low-pass filter with a roll-off of 6dB/octave. The cutoff frequency will be attenuated by -3dB (i.e.: 0.707x as loud).

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.MP3Compressor`
An MP3 compressor plugin that runs the LAME MP3 encoder in real-time to add compression artifacts to the audio stream.

Currently only supports variable bit-rate mode (VBR) and accepts a floating-point VBR quality value (between 0.0 and 10.0; lower is better).

Note that the MP3 format only supports 8kHz, 11025Hz, 12kHz, 16kHz, 22050Hz, 24kHz, 32kHz, 44.1kHz, and 48kHz audio; if an unsupported sample rate is provided, an exception will be thrown at processing time.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Mix`
A utility plugin that allows running other plugins in parallel. All plugins provided will be mixed equally.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `append(plugin: Plugin) -> None` – Append a plugin to the end of this container.
- `insert(index: int, plugin: Plugin) -> None` – Insert a plugin at the specified index.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `remove(plugin: Plugin) -> None` – Remove a plugin by its value.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.NoiseGate`
A simple noise gate with standard threshold, ratio, attack time and release time controls. Can be used as an expander if the ratio is low.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.PeakFilter`
A peak (or notch) filter with variable Q and gain, as would be used in an equalizer. Frequencies around the cutoff frequency will be boosted (or cut) by the provided gain value.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Phaser`
A 6 stage phaser that modulates first order all-pass filters to create sweeping notches in the magnitude frequency response. This audio effect can be controlled with standard phaser parameters: the speed and depth of the LFO controlling the frequency response, a mix control, a feedback control, and the centre frequency of the modulation.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.PitchShift`
A pitch shifting effect that can change the pitch of audio without affecting its duration.

This effect uses Chris Cannam’s wonderful *Rubber Band* library audio stretching library.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Plugin`
A generic audio processing plugin. Base class of all Pedalboard plugins.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.PluginContainer`
A generic audio processing plugin that contains zero or more other plugins. Not intended for direct use.

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `append(plugin: Plugin) -> None` – Append a plugin to the end of this container.
- `insert(index: int, plugin: Plugin) -> None` – Insert a plugin at the specified index.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `remove(plugin: Plugin) -> None` – Remove a plugin by its value.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.Resample`
A plugin that downsamples the input audio to the given sample rate, then upsamples it back to the original sample rate. Various quality settings will produce audible distortion and aliasing effects.

#### Class `Quality`
Indicates which specific resampling algorithm to use.

Resampling algorithms each provide a different tradeoff between speed and quality. Pedalboard provides two different types of resampling algorithms:

- **Aliasing algorithms**, which cause high frequencies to appear as lower frequencies.
- **Non-aliasing algorithms**, which filter out high frequencies when downsampling and avoid introducing extra high-frequency content when upsampling. (These algorithms were introduced in Pedalboard v0.9.15.)

Aliasing algorithms include:
- `ZeroOrderHold`
- `Linear`
- `CatmullRom`
- `Lagrange`
- `WindowedSinc`

Non-aliasing algorithms include:
- `WindowedSinc256`
- `WindowedSinc128`
- `WindowedSinc64`
- `WindowedSinc32`
- `WindowedSinc16`
- `WindowedSinc8`

Choosing an algorithm to use depends on the signal being resampled, the relationship between the source and target sample rates, and the application of the resampled signal.

If downsampling by an integer factor (i.e.: from 44.1kHz to 22050Hz, or 48kHz to 24kHz), and if the source signal has no high-frequency content above half of the target sample rate the `ZeroOrderHold` algorithm will be the fastest by far and will produce no artifacts.

In all other cases, any of the numbered `WindowedSinc` algorithms (i.e.: `WindowedSinc256`, `WindowedSinc64`) will produce a clean signal with no artifacts. Higher numbers will produce a cleaner signal with less roll-off of high frequency content near the Nyquist frequency of the new sample rate.

However, depending on your application, the artifacts introduced by each resampling method may be acceptable. Test each method to determine which is the best tradeoff between speed and accuracy for your use case.

To provide a good balance between speed and accuracy, `WindowedSinc32` is the default from Pedalboard v0.9.15 onwards. (Previously, `WindowedSinc` was the default.)

##### Members
- `ZeroOrderHold` – The lowest quality and fastest resampling method, with lots of audible artifacts. Zero-order hold resampling chooses the next value to use based on the last value, without any interpolation. Think of it like nearest-neighbor resampling. **Warning:** This algorithm produces aliasing artifacts.
- `Linear` – A resampling method slightly less noisy than the simplest method. Linear resampling takes the average of the two nearest values to the desired sample, which is reasonably good for downsampling. **Warning:** This algorithm produces aliasing artifacts.
- `CatmullRom` – A moderately good-sounding resampling method which is fast to run. Slightly slower than Linear resampling, but slightly higher quality. **Warning:** This algorithm produces aliasing artifacts.
- `Lagrange` – A moderately good-sounding resampling method which is slow to run. Slower than CatmullRom resampling, but slightly higher quality. **Warning:** This algorithm produces aliasing artifacts.
- `WindowedSinc` – A very high quality (and the slowest) resampling method, with no audible artifacts when upsampling. This resampler applies a windowed sinc filter design with 100 zero-crossings of the sinc function to approximate an ideal brick-wall low-pass filter. **Warning:** This algorithm produces aliasing artifacts when downsampling, but not when upsampling. **Note:** This method was the default in versions of Pedalboard prior to v0.9.15.
- `WindowedSinc256` – The highest possible quality resampling algorithm, with no audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 256 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter does not produce aliasing artifacts when upsampling or downsampling. Compare this in speed and quality to Resampy’s `kaiser_best` method.
- `WindowedSinc128` – A very high quality resampling algorithm, with no audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 128 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter does not produce aliasing artifacts when upsampling or downsampling. This method is roughly as fast as Resampy’s `kaiser_fast` method, while producing results roughly equal in quality to Resampy’s `kaiser_best` method.
- `WindowedSinc64` – A very high quality resampling algorithm, with few audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 64 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter does not produce aliasing artifacts when upsampling or downsampling. This method is (on average) faster than Resampy’s `kaiser_fast` method, and roughly equal in quality.
- `WindowedSinc32` – A reasonably high quality resampling algorithm, with few audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 32 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter produces very few aliasing artifacts when upsampling or downsampling. This method is always faster than Resampy’s `kaiser_fast` method, while being reasonable in quality. **Note:** This method is the default in Pedalboard v0.9.15 and later.
- `WindowedSinc16` – A medium quality resampling algorithm, with few audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 16 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter produces some aliasing artifacts when upsampling or downsampling. This method is faster than Resampy’s `kaiser_fast` method, while being acceptable in quality.
- `WindowedSinc8` – A low quality resampling algorithm, with few audible artifacts when upsampling or downsampling. This resampler applies a windowed sinc filter with 16 zero-crossings to approximate an ideal brick-wall low-pass filter. This filter produces noticeable aliasing artifacts when upsampling or downsampling. This method can be more than 10x faster than Resampy’s `kaiser_fast` method, and is useful for applications that are tolerant of some resampling artifacts.

##### Properties
- `name`

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.
- `quality` – The resampling algorithm used to resample the audio.
- `target_sample_rate` – The sample rate to resample the input audio to. This value may be a floating-point number, in which case a floating-point sampling rate will be used. Note that the output of this plugin will still be at the original sample rate; this is merely the sample rate used for quality reduction.

---

### `pedalboard.Reverb`
A simple reverb effect. Uses a simple stereo reverb algorithm, based on the technique and tunings used in [FreeVerb](https://ccrma.stanford.edu/~jos/pasp/Freeverb.html).

#### Methods
- `__call__(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio buffer through this plugin. Alias for `process()`.
- `process(input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run a 32-bit or 64-bit floating point audio buffer through this plugin.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.

#### Properties
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin is not an audio effect and accepts only MIDI input, not audio. Introduced in v0.7.4.

---

### `pedalboard.VST3Plugin`
A wrapper around third-party, audio effect or instrument plugins in Steinberg GmbH’s VST3® format.

VST3® plugins are supported on macOS, Windows, and Linux. However, VST3® plugin files are not cross-compatible with different operating systems; a platform-specific build of each plugin is required to load that plugin on a given platform. (For example: a Windows VST3 plugin bundle will not load on Linux or macOS.)

> **Warning:** Some VST3® plugins may throw errors, hang, generate incorrect output, or outright crash if called from background threads. If you find that a VST3® plugin is not working as expected, try calling it from the main thread instead and open a GitHub Issue to track the incompatibility.

Support for instrument plugins introduced in v0.7.4.  
Support for running VST3® plugins on background threads introduced in v0.8.8.

#### Methods
- `__call__(*args, **kwargs)` – Overloaded function.
  - `__call__(self: pedalboard.Plugin, input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio or MIDI buffer through this plugin, returning audio. Alias for `process()`.
  - `__call__(self: pedalboard.VST3Plugin, midi_messages: object, duration: float, sample_rate: float, num_channels: int = 2, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Run an audio or MIDI buffer through this plugin, returning audio. Alias for `process()`.
- `load_preset(preset_file_path: str) -> None` – Load a VST3 preset file in `.vstpreset` format.
- `process(*args, **kwargs)` – Overloaded function.
  - `process(self: pedalboard.Plugin, input_array: numpy.ndarray, sample_rate: float, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Pass a buffer of audio (as a 32- or 64-bit NumPy array) or a list of MIDI messages to this plugin, returning audio.
  - `process(self: pedalboard.VST3Plugin, midi_messages: object, duration: float, sample_rate: float, num_channels: int = 2, buffer_size: int = 8192, reset: bool = True) -> numpy.ndarray[numpy.float32]` – Pass a buffer of audio (as a 32- or 64-bit NumPy array) or a list of MIDI messages to this plugin, returning audio.
- `reset() -> None` – Clear any internal state stored by this plugin (e.g.: reverb tails, delay lines, LFO state, etc). The values of plugin parameters will remain unchanged.
- `show_editor(close_event: object = None) -> None` – Show the UI of this plugin as a native window.  
  This method may only be called on the main thread, and will block the main thread until any of the following things happens:
  - the window is closed by clicking the close button
  - the window is closed by pressing the appropriate (OS-specific) keyboard shortcut
  - a KeyboardInterrupt (Ctrl-C) is sent to the program
  - the `threading.Event.set()` method is called (by another thread) on a provided `threading.Event` object

  An example of how to programmatically close an editor window:
  ```python
  import pedalboard
  from threading import Event, Thread

  plugin = pedalboard.load_plugin("../path-to-my-plugin-file")
  close_window_event = Event()

  def other_thread():
      # do something to determine when to close the window
      if should_close_window:
          close_window_event.set()

  thread = Thread(target=other_thread)
  thread.start()

  # This will block until the other thread calls .set():
  plugin.show_editor(close_window_event)
  ```

#### Properties
- `category` – A category that this plugin falls into, such as "Dynamics", "Reverbs", etc. Introduced in v0.9.4.
- `descriptive_name` – A more descriptive name for this plugin. This may be the same as the ‘name’ field, but some plugins may provide an alternative name. Introduced in v0.9.4.
- `has_shared_container` – True iff this plugin is part of a multi-plugin container. Introduced in v0.9.4.
- `identifier` – A string that can be saved and used to uniquely identify this plugin (and version) again. Introduced in v0.9.4.
- `is_effect` – True iff this plugin is an audio effect and accepts audio as input. Introduced in v0.7.4.
- `is_instrument` – True iff this plugin identifies itself as an instrument (generator, synthesizer, etc) plugin. Introduced in v0.9.4.
- `manufacturer_name` – The name of the manufacturer of this plugin, as reported by the plugin itself. Introduced in v0.9.4.
- `name` – The name of this plugin.
- `preset_data` – Get or set the current plugin state as bytes in `.vstpreset` format.  
  > **Warning:** This property can be set to change the plugin’s internal state, but providing invalid data may cause the plugin to crash, taking the entire Python process down with it.
- `raw_state` – A bytes object representing the plugin’s internal state. For the VST3 format, this is usually an XML-encoded string prefixed with an 8-byte header and suffixed with a single null byte.  
  > **Warning:** This property can be set to change the plugin’s internal state, but providing invalid data may cause the plugin to crash, taking the entire Python process down with it.
- `reported_latency_samples` – The number of samples of latency (delay) that this plugin reports to introduce into the audio signal due to internal buffering and processing. Pedalboard automatically compensates for this latency during processing, so this property is present for informational purposes. Note that not all plugins correctly report the latency that they introduce, so this value may be inaccurate (especially if the plugin reports 0). Introduced in v0.9.12.
- `version` – The version string for this plugin, as reported by the plugin itself. Introduced in v0.9.4.

---

## Functions

### `time_stretch`

```python
def time_stretch(
    input_audio: numpy.ndarray[numpy.float32],
    samplerate: float,
    stretch_factor: Union[float, numpy.ndarray[numpy.float64]] = 1.0,
    pitch_shift_in_semitones: Union[float, numpy.ndarray[numpy.float64]] = 0.0,
    high_quality: bool = True,
    transient_mode: str = "crisp",
    transient_detector: str = "compound",
    retain_phase_continuity: bool = True,
    use_long_fft_window: Optional[bool] = None,
    use_time_domain_smoothing: bool = False,
    preserve_formants: bool = True,
) -> numpy.ndarray[numpy.float32]
```

Time-stretch (and optionally pitch-shift) a buffer of audio, changing its length.

Using a higher `stretch_factor` will shorten the audio – i.e., a `stretch_factor` of 2.0 will double the speed of the audio and halve the length of the audio, without changing the pitch of the audio.

This function allows for changing the pitch of the audio during the time stretching operation. The `stretch_factor` and `pitch_shift_in_semitones` arguments are independent and do not affect each other (i.e.: you can change one, the other, or both without worrying about how they interact).

Both `stretch_factor` and `pitch_shift_in_semitones` can be either floating-point numbers or NumPy arrays of double-precision floating point numbers. Providing a NumPy array allows the stretch factor and/or pitch shift to vary over the length of the output audio.

> **Note:** If a NumPy array is provided for `stretch_factor` or `pitch_shift_in_semitones`:
> - The length of each array must be the same as the length of the input audio.
> - More frequent changes in the stretch factor or pitch shift will result in slower processing, as the audio will be processed in smaller chunks.
> - Changes to the `stretch_factor` or `pitch_shift_in_semitones` more frequent than once every 1,024 samples (23 milliseconds at 44.1kHz) will not have any effect.

The additional arguments provided to this function allow for more fine-grained control over the behavior of the time stretcher:

- `high_quality` (the default) enables a higher quality time stretching mode. Set this option to `False` to use less CPU power.
- `transient_mode` controls the behavior of the stretcher around transients (percussive parts of the audio). Valid options are `"crisp"` (the default), `"mixed"`, or `"smooth"`.
- `transient_detector` controls which method is used to detect transients in the audio signal. Valid options are `"compound"` (the default), `"percussive"`, or `"soft"`.
- `retain_phase_continuity` ensures that the phases of adjacent frequency bins in the audio stream are kept as similar as possible. Set this to `False` for a softer, phasier sound.
- `use_long_fft_window` controls the size of the fast-Fourier transform window used during stretching. The default (`None`) will result in a window size that varies based on other parameters and should produce better results in most situations. Set this option to `True` to result in a smoother sound (at the expense of clarity and timing), or `False` to result in a crisper sound.
- `use_time_domain_smoothing` can be enabled to produce a softer sound with audible artifacts around sharp transients. This option mixes well with `use_long_fft_window=False`.
- `preserve_formants` allows shifting the pitch of notes without substantially affecting the pitch profile (formants) of a voice or instrument.

> **Warning:** This is a function, not a `Plugin` instance, and cannot be used in `Pedalboard` objects, as it changes the duration of the audio stream.

> **Note:** The ability to pass a NumPy array for `stretch_factor` and `pitch_shift_in_semitones` was added in Pedalboard v0.9.8.