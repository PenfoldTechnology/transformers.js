
/**
 * @file Processors are used to prepare non-textual inputs (e.g., image or audio) for a model.
 * 
 * **Example:** Using a `WhisperProcessor` to prepare an audio input for a model.
 * ```javascript
 * import { AutoProcessor, read_audio } from '@xenova/transformers';
 *
 * let processor = await AutoProcessor.from_pretrained('openai/whisper-tiny.en');
 * let audio = await read_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
 * let { input_features } = await processor(audio);
 * // Tensor {
 * //   data: Float32Array(240000) [0.4752984642982483, 0.5597258806228638, 0.56434166431427, ...],
 * //   dims: [1, 80, 3000],
 * //   type: 'float32',
 * //   size: 240000,
 * // }
 * ```
 * 
 * @module processors
 */
import {
    Callable,
    calculateDimensions,
    calculateReflectOffset,
} from './utils/core.js';

import {
    getModelJSON,
} from './utils/hub.js';

import {
    min,
    max,
    softmax,
    bankers_round,
} from './utils/maths.js';


import { Tensor, permute, cat, interpolate, stack } from './utils/tensor.js';

import {
    window_function,
    spectrogram,
    mel_filter_bank,
} from './utils/audio.js';


// Helper functions

/**
 * Converts bounding boxes from center format to corners format.
 * 
 * @param {number[]} arr The coordinate for the center of the box and its width, height dimensions (center_x, center_y, width, height)
 * @returns {number[]} The coodinates for the top-left and bottom-right corners of the box (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
 */
function center_to_corners_format([centerX, centerY, width, height]) {
    return [
        centerX - width / 2,
        centerY - height / 2,
        centerX + width / 2,
        centerY + height / 2
    ];
}

/**
 * Post-processes the outputs of the model (for object detection).
 * @param {Object} outputs The outputs of the model that must be post-processed
 * @param {Tensor} outputs.logits The logits
 * @param {Tensor} outputs.pred_boxes The predicted boxes.
 * @param {number} [threshold=0.5] The threshold to use for the scores.
 * @param {number[][]} [target_sizes=null] The sizes of the original images.
 * @param {boolean} [is_zero_shot=false] Whether zero-shot object detection was performed.
 * @return {Object[]} An array of objects containing the post-processed outputs.
 * @private
 */
function post_process_object_detection(outputs, threshold = 0.5, target_sizes = null, is_zero_shot = false) {
    const out_logits = outputs.logits;
    const out_bbox = outputs.pred_boxes;
    const [batch_size, num_boxes, num_classes] = out_logits.dims;

    if (target_sizes !== null && target_sizes.length !== batch_size) {
        throw Error("Make sure that you pass in as many target sizes as the batch dimension of the logits")
    }
    let toReturn = [];
    for (let i = 0; i < batch_size; ++i) {
        let target_size = target_sizes !== null ? target_sizes[i] : null;
        let info = {
            boxes: [],
            classes: [],
            scores: []
        }
        let logits = out_logits[i];
        let bbox = out_bbox[i];

        for (let j = 0; j < num_boxes; ++j) {
            let logit = logits[j];

            let indices = [];
            let probs;
            if (is_zero_shot) {
                // Get indices of classes with high enough probability
                probs = logit.sigmoid().data;
                for (let k = 0; k < probs.length; ++k) {
                    if (probs[k] > threshold) {
                        indices.push(k);
                    }
                }

            } else {
                // Get most probable class
                let maxIndex = max(logit.data)[1];

                if (maxIndex === num_classes - 1) {
                    // This is the background class, skip it
                    continue;
                }
                indices.push(maxIndex);

                // Compute softmax over classes
                probs = softmax(logit.data);
            }

            for (const index of indices) {

                // Some class has a high enough probability
                /** @type {number[]} */
                let box = bbox[j].data;

                // convert to [x0, y0, x1, y1] format
                box = center_to_corners_format(box)
                if (target_size !== null) {
                    box = box.map((x, i) => x * target_size[(i + 1) % 2])
                }

                info.boxes.push(box);
                info.classes.push(index);
                info.scores.push(probs[index]);
            }
        }
        toReturn.push(info);
    }
    return toReturn;
}

/**
 * Named tuple to indicate the order we are using is (height x width), even though
 * the Graphics’ industry standard is (width x height).
 * @typedef {[height: number, width: number]} HeightWidth
 */

/**
 * Helper function to validate audio inputs.
 * @param {any} audio The audio data.
 * @param {string} feature_extractor The name of the feature extractor.
 * @private
 */
function validate_audio_inputs(audio, feature_extractor) {
    if (!(audio instanceof Float32Array || audio instanceof Float64Array)) {
        throw new Error(
            `${feature_extractor} expects input to be a Float32Array or a Float64Array, but got ${audio?.constructor?.name ?? typeof audio} instead. ` +
            `If using the feature extractor directly, remember to use \`read_audio(url, sampling_rate)\` to obtain the raw audio data of the file/url.`
        )
    }
}

/**
 * Helper function to constrain a value to be a multiple of a number.
 * @param {number} val The value to constrain.
 * @param {number} multiple The number to constrain to.
 * @param {number} [minVal=0] The minimum value to constrain to.
 * @param {number} [maxVal=null] The maximum value to constrain to.
 * @returns {number} The constrained value.
 * @private
 */
function constraint_to_multiple_of(val, multiple, minVal = 0, maxVal = null) {
    const a = val / multiple;
    let x = bankers_round(a) * multiple;

    if (maxVal !== null && x > maxVal) {
        x = Math.floor(a) * multiple;
    }

    if (x < minVal) {
        x = Math.ceil(a) * multiple;
    }

    return x;
}

/**
 * Rounds the height and width down to the closest multiple of size_divisibility
 * @param {[number, number]} size The size of the image
 * @param {number} divisor The divisor to use.
 * @returns {[number, number]} The rounded size.
 */
function enforce_size_divisibility([width, height], divisor) {
    return [
        Math.max(Math.floor(width / divisor), 1) * divisor,
        Math.max(Math.floor(height / divisor), 1) * divisor
    ];
}


/**
 * Base class for feature extractors.
 *
 * @extends Callable
 */
export class FeatureExtractor extends Callable {
    /**
     * Constructs a new FeatureExtractor instance.
     *
     * @param {Object} config The configuration for the feature extractor.
     */
    constructor(config) {
        super();
        this.config = config
    }
}
export class WhisperFeatureExtractor extends FeatureExtractor {

    constructor(config) {
        super(config);

        // Prefer given `mel_filters` from preprocessor_config.json, or calculate them if they don't exist.
        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + this.config.n_fft / 2), // num_frequency_bins
            this.config.feature_size, // num_mel_filters
            0.0, // min_frequency
            8000.0, // max_frequency
            this.config.sampling_rate, // sampling_rate
            "slaney", // norm
            "slaney", // mel_scale
        );

        this.window = window_function(this.config.n_fft, 'hann');
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {{data: Float32Array, dims: number[]}} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    _extract_fbank_features(waveform) {
        const { data, dims } = spectrogram(
            waveform,
            this.window, // window
            this.config.n_fft, // frame_length
            this.config.hop_length, // hop_length
            {
                power: 2.0,
                mel_filters: this.config.mel_filters,
                log_mel: 'log10',

                // Custom
                max_num_frames: this.config.nb_max_frames, // 3000
            }
        )

        const maxValue = max(data)[0];

        for (let i = 0; i < data.length; ++i) {
            data[i] = (Math.max(data[i], maxValue - 8.0) + 4.0) / 4.0;
        }

        return { data, dims };
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'WhisperFeatureExtractor');

        let waveform;
        if (audio.length > this.config.n_samples) {
            console.warn(
                "Attempting to extract features for audio longer than 30 seconds. " +
                "If using a pipeline to extract transcript from a long audio clip, " +
                "remember to specify `chunk_length_s` and/or `stride_length_s`."
            );
            waveform = audio.slice(0, this.config.n_samples);
        } else {
            // pad with zeros
            waveform = new Float32Array(this.config.n_samples);
            waveform.set(audio);
        }

        const { data, dims } = this._extract_fbank_features(waveform);

        return {
            input_features: new Tensor('float32',
                data,
                [1, ...dims]
            )
        };
    }
}

export class Wav2Vec2FeatureExtractor extends FeatureExtractor {

    /**
     * @param {Float32Array} input_values 
     * @returns {Float32Array} 
     */
    _zero_mean_unit_var_norm(input_values) {
        // TODO support batch?
        const sum = input_values.reduce((a, b) => a + b, 0);
        const mean = sum / input_values.length;
        const variance = input_values.reduce((a, b) => a + (b - mean) ** 2, 0) / input_values.length;
        return input_values.map(x => (x - mean) / Math.sqrt(variance + 1e-7));
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_values: Tensor; attention_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention mask as Tensors.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'Wav2Vec2FeatureExtractor');

        if (audio instanceof Float64Array) {
            audio = new Float32Array(audio);
        }

        let input_values = audio;

        // zero-mean and unit-variance normalization
        if (this.config.do_normalize) {
            input_values = this._zero_mean_unit_var_norm(input_values);
        }

        // TODO: allow user to pass in attention mask
        const shape = [1, input_values.length];
        return {
            input_values: new Tensor('float32', input_values, shape),
            attention_mask: new Tensor('int64', new BigInt64Array(input_values.length).fill(1n), shape)
        };
    }
}

export class SeamlessM4TFeatureExtractor extends FeatureExtractor {

    constructor(config) {
        super(config);

        const sampling_rate = this.config.sampling_rate;
        const mel_filters = mel_filter_bank(
            256, // num_frequency_bins
            this.config.num_mel_bins, // num_mel_filters
            20, // min_frequency
            Math.floor(sampling_rate / 2), // max_frequency
            sampling_rate, // sampling_rate
            null, // norm
            "kaldi", // mel_scale
            true, // triangularize_in_mel_space
        );

        // Do padding:
        for (let i = 0; i < mel_filters.length; ++i) {
            mel_filters[i].push(0);
        }
        this.mel_filters = mel_filters;

        this.window = window_function(400, 'povey', {
            periodic: false,
        })
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @param {number} max_length The maximum number of frames to return.
     * @returns {{data: Float32Array, dims: number[]}} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    _extract_fbank_features(waveform, max_length) {
        // NOTE: We don't pad/truncate since that is passed in as `max_num_frames`

        // Kaldi compliance: 16-bit signed integers
        // 32768 == 2 ** 15
        waveform = waveform.map((/** @type {number} */ x) => x * 32768)

        return spectrogram(
            waveform,
            this.window, // window
            400, // frame_length
            160, // hop_length
            {
                fft_length: 512,
                power: 2.0,
                center: false,
                preemphasis: 0.97,
                mel_filters: this.mel_filters,
                log_mel: 'log',
                mel_floor: 1.192092955078125e-07,
                remove_dc_offset: true,

                // Custom
                max_num_frames: max_length,
                transpose: true,
            }
        )
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @param {Object} options Optional parameters for feature extraction.
     * @param {boolean} [options.padding=true] Whether to pad the sequence to a multiple of `pad_to_multiple_of`.
     * @param {number} [options.pad_to_multiple_of=2] The number to pad the sequence to a multiple of.
     * @param {boolean} [options.do_normalize_per_mel_bins=true] Whether or not to zero-mean unit-variance normalize the input per mel-channel.
     * @param {boolean} [options.return_attention_mask=true] Whether to return the attention mask.
     * @returns {Promise<{ input_features: Tensor, attention_mask?: Tensor }>} A Promise resolving to an object containing the extracted input features and attention masks as Tensors.
     */
    async _call(audio, {
        padding = true,
        pad_to_multiple_of = 2,
        do_normalize_per_mel_bins = true,
        return_attention_mask = true,
    } = {}) {
        validate_audio_inputs(audio, 'SeamlessM4TFeatureExtractor');

        let features = this._extract_fbank_features(audio, this.config.max_length);

        if (do_normalize_per_mel_bins) {
            const [num_features, feature_size] = features.dims;
            for (let i = 0; i < feature_size; ++i) {
                let sum = 0;
                for (let j = 0; j < num_features; ++j) {
                    sum += features.data[j * feature_size + i];
                }

                const mean = sum / num_features;

                let variance = 0;
                for (let j = 0; j < num_features; ++j) {
                    variance += (features.data[j * feature_size + i] - mean) ** 2;
                }
                variance /= num_features - 1; // NOTE: We use ddof=1

                const std = Math.sqrt(variance + 1e-7);
                for (let j = 0; j < num_features; ++j) {
                    const index = j * feature_size + i;
                    features.data[index] = (features.data[index] - mean) / std;
                }
            }
        }

        let padded_attention_mask;
        if (padding) {
            const [num_frames, num_channels] = features.dims;

            const pad_size = num_frames % pad_to_multiple_of;
            if (pad_size > 0) {
                const padded_data = new Float32Array(num_channels * (num_frames + pad_size));
                padded_data.set(features.data)
                padded_data.fill(this.config.padding_value, features.data.length)

                const numPaddedFrames = num_frames + pad_size;
                features = {
                    data: padded_data,
                    dims: [numPaddedFrames, num_channels],
                }

                if (return_attention_mask) {
                    padded_attention_mask = new Tensor(
                        'int64',
                        new BigInt64Array(numPaddedFrames),
                        [1, numPaddedFrames],
                    )
                    padded_attention_mask.data.fill(1n, 0, num_frames);
                }
            }
        }

        const [num_frames, num_channels] = features.dims;

        const stride = this.config.stride;
        const remainder = num_frames % stride;
        if (remainder !== 0) {
            throw new Error(`The number of frames (${num_frames}) must be a multiple of the stride (${stride}).`)
        }

        const input_features = new Tensor('float32',
            features.data,
            features.dims,
        ).view(
            1,
            Math.floor(num_frames / stride),
            num_channels * stride,
        );

        const result = { input_features }

        if (return_attention_mask) {
            const reshapedNumFrames = input_features.dims[1];

            const attention_mask = new Tensor(
                'int64',
                new BigInt64Array(reshapedNumFrames),
                [1, reshapedNumFrames],
            );
            if (padded_attention_mask) {
                for (let i = 1, j = 0; i < num_frames; i += stride, ++j) {
                    attention_mask.data[j] = padded_attention_mask.data[i];
                }
            } else {
                attention_mask.data.fill(1n);
            }

            result.attention_mask = attention_mask;
        }

        return result;
    }
}

export class ASTFeatureExtractor extends FeatureExtractor {


    constructor(config) {
        super(config);

        const sampling_rate = this.config.sampling_rate;
        const mel_filters = mel_filter_bank(
            256, // num_frequency_bins
            this.config.num_mel_bins, // num_mel_filters
            20, // min_frequency
            Math.floor(sampling_rate / 2), // max_frequency
            sampling_rate, // sampling_rate
            null, // norm
            "kaldi", // mel_scale
            true, // triangularize_in_mel_space
        );

        // Do padding:
        for (let i = 0; i < mel_filters.length; ++i) {
            mel_filters[i].push(0);
        }
        this.mel_filters = mel_filters;

        this.window = window_function(400, 'hann', {
            periodic: false,
        })

        this.mean = this.config.mean;
        this.std = this.config.std;
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @param {number} max_length The maximum number of frames to return.
     * @returns {{data: Float32Array, dims: number[]}} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    _extract_fbank_features(waveform, max_length) {
        // NOTE: We don't pad/truncate since that is passed in as `max_num_frames`
        return spectrogram(
            waveform,
            this.window, // window
            400, // frame_length
            160, // hop_length
            {
                fft_length: 512,
                power: 2.0,
                center: false,
                preemphasis: 0.97,
                mel_filters: this.mel_filters,
                log_mel: 'log',
                mel_floor: 1.192092955078125e-07,
                remove_dc_offset: true,

                // Custom
                max_num_frames: max_length,
                transpose: true,
            }
        )
    }


    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_values: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'ASTFeatureExtractor');

        const features = this._extract_fbank_features(audio, this.config.max_length);
        if (this.config.do_normalize) {
            // Normalize the input audio spectrogram to have mean=0, std=0.5
            const denom = this.std * 2;
            for (let i = 0; i < features.data.length; ++i) {
                features.data[i] = (features.data[i] - this.mean) / denom;
            }
        }

        return {
            input_values: new Tensor('float32',
                features.data,
                [1, ...features.dims]
            )
        };
    }
}

export class ClapFeatureExtractor extends FeatureExtractor {

    constructor(config) {
        super(config);

        this.mel_filters = mel_filter_bank(
            this.config.nb_frequency_bins, // num_frequency_bins
            this.config.feature_size, // num_mel_filters
            this.config.frequency_min, // min_frequency
            this.config.frequency_max, // max_frequency
            this.config.sampling_rate, // sampling_rate
            null, // norm
            "htk", // mel_scale
        );

        this.mel_filters_slaney = mel_filter_bank(
            this.config.nb_frequency_bins, // num_frequency_bins
            this.config.feature_size, // num_mel_filters
            this.config.frequency_min, // min_frequency
            this.config.frequency_max, // max_frequency
            this.config.sampling_rate, // sampling_rate
            "slaney", // norm
            "slaney", // mel_scale
        );

        this.window = window_function(this.config.fft_window_size, 'hann')

    }


    /**
     * Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.
     * 
     * Four different path are possible:
     *   - `truncation="fusion"` and the length of the waveform is greater than the max length: the mel spectrogram
     *     will be computed on the entire audio. 3 random crops and a dowsampled version of the full mel spectrogram
     *     are then stacked together. They will later be used for `feature_fusion`.
     *   - `truncation="rand_trunc"` and the length of the waveform is smaller than the max length: the audio is
     *     padded based on `padding`.
     *   - `truncation="fusion"` and the length of the waveform is smaller than the max length: the audio is padded
     *     based on `padding`, and is repeated `4` times.
     *   - `truncation="rand_trunc"` and the length of the waveform is greater than the max length: the mel
     *     spectrogram will be computed on a random crop of the waveform.
     * 
     * @param {Float32Array|Float64Array} waveform The input waveform.
     * @param {number} max_length The maximum length of the waveform.
     * @param {string} truncation The truncation strategy to use.
     * @param {string} padding The padding strategy to use.
     * @returns {{ data: Float32Array; dims: number[]; longer: boolean; }} An object containing the mel spectrogram data as a Float32Array, its dimensions as an array of numbers, and a boolean indicating whether the waveform was longer than the max length.
     */
    _get_input_mel(waveform, max_length, truncation, padding) {

        /** @type {{ data: Float32Array; dims: number[]}} */
        let input_mel;
        let longer = false;
        const diff = waveform.length - max_length;
        if (diff > 0) {
            if (truncation === 'rand_trunc') {
                longer = true;
                const idx = Math.floor(Math.random() * (diff + 1));
                waveform = waveform.subarray(idx, idx + max_length);

                input_mel = this._extract_fbank_features(waveform, this.mel_filters_slaney, this.config.nb_max_samples);
                input_mel.dims = [1, ...input_mel.dims]; // "unsqueeze"
            } else {
                // TODO implement fusion strategy
                throw new Error(`Truncation strategy "${truncation}" not implemented`)
            }
        } else {
            if (diff < 0) {
                let padded = new Float64Array(max_length); // already padded with zeros
                padded.set(waveform);

                if (padding === 'repeat') {
                    for (let i = waveform.length; i < max_length; i += waveform.length) {
                        padded.set(waveform.subarray(0, Math.min(waveform.length, max_length - i)), i);
                    }
                } else if (padding === 'repeatpad') {
                    for (let i = waveform.length; i < -diff; i += waveform.length) {
                        padded.set(waveform, i);
                    }
                }
                waveform = padded;
            }

            if (truncation === 'fusion') {
                throw new Error(`Truncation strategy "${truncation}" not implemented`)
            }

            input_mel = this._extract_fbank_features(waveform, this.mel_filters_slaney, this.config.nb_max_samples);
            input_mel.dims = [1, ...input_mel.dims]; // "unsqueeze"
        }

        return {
            ...input_mel,
            longer,
        }
    }

    /**
     * Compute the log-mel spectrogram of the provided `waveform` using the Hann window.
     * In CLAP, two different filter banks are used depending on the truncation pattern:
     *  - `self.mel_filters`: they correspond to the default parameters of `torchaudio` which can be obtained from
     *    calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`
     *    is set to `"fusion"`.
     *  - `self.mel_filteres_slaney` : they correspond to the default parameters of `librosa` which used
     *    `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original
     *    implementation when the truncation mode is not `"fusion"`.
     * 
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @param {number[][]} mel_filters The mel filters to use.
     * @param {number} [max_length=null] The maximum number of frames to return.
     * @returns {{data: Float32Array, dims: number[]}} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    _extract_fbank_features(waveform, mel_filters, max_length = null) {
        // NOTE: We don't pad/truncate since that is passed in as `max_num_frames`
        return spectrogram(
            waveform,
            this.window, // window
            this.config.fft_window_size, // frame_length
            this.config.hop_length, // hop_length
            {
                power: 2.0,
                mel_filters,
                log_mel: 'dB',

                // Custom
                max_num_frames: max_length,
                do_pad: false,
                transpose: true,
            }
        )
    }


    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio, {
        max_length = null,
    } = {}) {
        validate_audio_inputs(audio, 'ClapFeatureExtractor');

        // convert to mel spectrogram, truncate and pad if needed.
        const padded_inputs = this._get_input_mel(
            audio,
            max_length ?? this.config.nb_max_samples,
            this.config.truncation,
            this.config.padding,
        );


        return {
            input_features: new Tensor('float32',
                padded_inputs.data,
                [1, ...padded_inputs.dims]
            )
        };
    }
}



export class SpeechT5FeatureExtractor extends FeatureExtractor { }

/**
 * Represents a Processor that extracts features from an input.
 * @extends Callable
 */
export class Processor extends Callable {
    /**
     * Creates a new Processor with the given feature extractor.
     * @param {FeatureExtractor} feature_extractor The function used to extract features from the input.
     */
    constructor(feature_extractor) {
        super();
        this.feature_extractor = feature_extractor;
        // TODO use tokenizer here?
    }

    /**
     * Calls the feature_extractor function with the given input.
     * @param {any} input The input to extract features from.
     * @param {...any} args Additional arguments.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(input, ...args) {
        return await this.feature_extractor(input, ...args);
    }
}

export class SamProcessor extends Processor {
    /**
     * @borrows SamImageProcessor#_call as _call
     */
    async _call(...args) {
        return await this.feature_extractor(...args);
    }

    /**
     * @borrows SamImageProcessor#post_process_masks as post_process_masks
     */
    post_process_masks(...args) {
        // @ts-ignore
        return this.feature_extractor.post_process_masks(...args);
    }
    /**
     * @borrows SamImageProcessor#reshape_input_points as reshape_input_points
     */
    reshape_input_points(...args) {
        // @ts-ignore
        return this.feature_extractor.reshape_input_points(...args);
    }
}

/**
 * Represents a WhisperProcessor that extracts features from an audio input.
 * @extends Processor
 */
export class WhisperProcessor extends Processor {
    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(audio) {
        return await this.feature_extractor(audio)
    }
}


export class Wav2Vec2ProcessorWithLM extends Processor {
    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(audio) {
        return await this.feature_extractor(audio)
    }
}

export class SpeechT5Processor extends Processor {
    /**
     * Calls the feature_extractor function with the given input.
     * @param {any} input The input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(input) {
        return await this.feature_extractor(input)
    }
}

export class OwlViTProcessor extends Processor { }


//////////////////////////////////////////////////
/**
 * Helper class which is used to instantiate pretrained processors with the `from_pretrained` function.
 * The chosen processor class is determined by the type specified in the processor config.
 * 
 * **Example:** Load a processor using `from_pretrained`.
 * ```javascript
 * let processor = await AutoProcessor.from_pretrained('openai/whisper-tiny.en');
 * ```
 * 
 * **Example:** Run an image through a processor.
 * ```javascript
 * let processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * let image_inputs = await processor(image);
 * // {
 * //   "pixel_values": {
 * //     "dims": [ 1, 3, 224, 224 ],
 * //     "type": "float32",
 * //     "data": Float32Array [ -1.558687686920166, -1.558687686920166, -1.5440893173217773, ... ],
 * //     "size": 150528
 * //   },
 * //   "original_sizes": [
 * //     [ 533, 800 ]
 * //   ],
 * //   "reshaped_input_sizes": [
 * //     [ 224, 224 ]
 * //   ]
 * // }
 * ```
 */
export class AutoProcessor {
    static FEATURE_EXTRACTOR_CLASS_MAPPING = {
        WhisperFeatureExtractor,
        Wav2Vec2FeatureExtractor,
        SeamlessM4TFeatureExtractor,
        SpeechT5FeatureExtractor,
        ASTFeatureExtractor,
        ClapFeatureExtractor,
    }

    static PROCESSOR_CLASS_MAPPING = {
        WhisperProcessor,
        Wav2Vec2ProcessorWithLM,
        SamProcessor,
        SpeechT5Processor,
        OwlViTProcessor,
    }

    /**
     * Instantiate one of the processor classes of the library from a pretrained model.
     * 
     * The processor class to instantiate is selected based on the `feature_extractor_type` property of the config object
     * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     * 
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
     * @param {import('./utils/hub.js').PretrainedOptions} options Additional options for loading the processor.
     * 
     * @returns {Promise<Processor>} A new instance of the Processor class.
     */
    static async from_pretrained(pretrained_model_name_or_path, {
        progress_callback = null,
        config = null,
        cache_dir = null,
        local_files_only = false,
        revision = 'main',
    } = {}) {

        let preprocessorConfig = config ?? await getModelJSON(pretrained_model_name_or_path, 'preprocessor_config.json', true, {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
        })

        // Determine feature extractor class
        // TODO: Ensure backwards compatibility with old configs
        let key = preprocessorConfig.feature_extractor_type ?? preprocessorConfig.image_processor_type;
        let feature_extractor_class = this.FEATURE_EXTRACTOR_CLASS_MAPPING[key];

        if (!feature_extractor_class) {
            throw new Error(`Unknown Feature Extractor type: ${key}`);
        }

        // If no associated processor class, use default
        let processor_class = this.PROCESSOR_CLASS_MAPPING[preprocessorConfig.processor_class] ?? Processor;

        // Instantiate processor and feature extractor
        let feature_extractor = new feature_extractor_class(preprocessorConfig);
        return new processor_class(feature_extractor);
    }
}
//////////////////////////////////////////////////

