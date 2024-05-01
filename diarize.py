import argparse
import os
from helpers import *
import whisperx
import torch
from transformers import pipeline
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
from transformers.utils import is_flash_attn_2_available

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="openai/whisper-medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs + [None],
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument(
    "--out-dir",
    dest="out_dir",
    default=None,
    help="Output directory for the results, defaults to the current directory",
)

args = parser.parse_args()

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file
pipe = pipeline(
    "automatic-speech-recognition",
    model=args.model_name,
    torch_dtype=torch.float16,
    device_map=args.device,
    model_kwargs={
        "attn_implementation": (
            "flash_attention_2" if is_flash_attn_2_available() else "eager"
        )
    },
)
# if args.suppress_numerals:
#     numeral_symbol_tokens = find_numeral_symbol_tokens(pipe.tokenizer)
#     token_list = [[token] for token in numeral_symbol_tokens[1:]]
# else:
#     token_list = None

whisper_results = pipe(
    vocal_target,
    chunk_length_s=30,
    batch_size=args.batch_size,
    generate_kwargs={
        "task": "transcribe",
        "language": args.language,
        # "bad_words_ids": token_list,
        "output_attentions": False,
    },
    return_timestamps=(True if args.language in wav2vec2_langs else "word"),
)

# convert audio to mono for NeMo combatibility
sound = AudioSegment.from_file(vocal_target).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

if args.language in wav2vec2_langs:
    segments = [
        {
            "text": chunk["text"],
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
        }
        for chunk in whisper_results["chunks"]
    ]
    if segments[-1]["end"] is None:
        segments[-1]["end"] = sound.duration_seconds
    alignment_model, metadata = whisperx.load_align_model(
        language_code=args.language, device=args.device
    )
    result_aligned = whisperx.align(
        segments, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=segments[0].get("start"),
        final_timestamp=segments[-1].get("end"),
    )
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    word_timestamps = [
        {
            "word": chunk["text"],
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
        }
        for chunk in whisper_results["chunks"]
    ]


# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if args.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {args.language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# write the results to a file in args.out_dir
if args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
