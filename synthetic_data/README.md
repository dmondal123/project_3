# Music Generation and MIDI Conversion

This engineering task involves generating music files using a recreation of OpenAI's MuseNet and converting them into MIDI files. You can input your custom MIDI file to generate n number of songs for each custom input provided. For recreating the genres in the Lakh MIDI Dataset used for the model, one song belonging to that genre is given as input to generate similar songs. For an example, Moondance by Van Morrison belongs to the genre "folk" according to the dataset which is further used to create 10 other songs sounding similar to Moondance.  Some of the generated songs are saved in the generated_songs folder. 

Note: This notebook can only be recreated on Google Colab if you are using a MAC M1 Pro Chip as it uses the library "fluidsynth" which does not work for this particular laptop. If you are on any other system, you can download fuildsynth as mentioned below.

## Some insights on the Songs Generated

The pre-trained model used to generate the songs, i.e. a [recreation of MuseNet](https://huggingface.co/hidude562/OpenMusenet-2.11-L) only has instruments as piano, drums, strings, woodwind, brass, and synthesizer which makes it difficult to recreate songs with instruents such as electric guitar or bass as its main instrument. However, it captures the essence of a rhythm guitar or bass which usually play a melody on loop throughout some sections of a song by recreating the same by playing multiple piano tracks simultaneously with one playing the main melody, the other playing the harmonies or chords and so on.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)

## Requirements

- Python 3.11 or later
- Required packages:
  - `mido`
  - `py_midicsv`
  - `fluidsynth`
  - `transformers`

You can install the required packages using pip:

```bash
pip install py_midicsv
pip install mido
pip install py_midicsv
apt install fluidsynth
cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2
pip install transformers
```

## Usage

- Reset MIDI Tracks: Before processing any files, ensure you reset the MIDI tracks using the reset_midi() function.
- Iterate over Generated Music Files: The code iterates over the generated music files (e.g., outAI_1.txt, outAI_2.txt, etc.) using a loop.
- Read and Parse Music Files: The code reads each generated music file and splits the content into individual events based on the pipe (|) character. It then parses each event by splitting it using commas.
- Sort Events: The events are sorted based on their timestamp to ensure the notes are played in the correct order.
- Convert Events into MIDI Notes: The code converts each event into MIDI notes. If an event represents a new note, it is added to a list of notes. If it represents the end of a note, the code calculates the duration and adds it to the list.
- Write MIDI Note Events: The MIDI note events are written to the MIDI tracks for each file.
- Add End Track Event: An end track event is added to the MIDI tracks to signify the end of the MIDI file.
- Write MIDI Files: Finally, the code writes the MIDI tracks to a MIDI file for each generated music file.