## What it measures

The rubric comes from the academy's own reviewer feedback on the sample classes. Reviewers
repeatedly praise or ask for the same things, grouped here into four teaching pillars plus a
camera check:

| Pillar | What earns a high score | Weight |
|---|---|---|
| Structure & rapport | Warm greeting, session intro, new joiners introduced, recap and goodbye | 25% |
| Individual attention | Children asked by name, everyone participates, understanding checked | 30% |
| Homework & reinforcement | Specific homework or practice set, key ideas recapped | 10% |
| Tone & energy | Lively, encouraging, specific praise, patient | 20% |
| Camera compliance | Tutor's face visible and well lit, camera on | gate |

## How a recording is graded

1. **Transcript.** A Zoom `.vtt` is used if provided. Otherwise the audio is transcribed with
   Whisper (Groq) and a small model labels each line as tutor or student.
2. **Pseudonymisation.** Speakers become *Tutor*, *S1*, *S2*…; children's names in the text
   become tags such as `[C3]` before anything is sent to a language model.
3. **Measured metrics.** Talk share, student turns per 10 minutes, questions, praise, names
   used, greeting / recap / homework detection, longest monologue. These are given to the
   judge as hints and shown in the report.
4. **Camera check, done locally.** One frame every 10 s is matched to whoever was speaking at
   that moment (Zoom records the active speaker). On the tutor's frames a local face detector
   (OpenCV YuNet) measures face visibility, camera-off time and lighting.
5. **Judge.** Claude Sonnet 5.5 scores each teaching pillar from 1 to 5 against fixed
   anchors and must cite `[mm:ss]` evidence for every score.
6. **Grade in code.** The weighted pillar score maps to a grade: ≥4.25 Excellent, ≥3.5 Good,
   ≥2.75 Average, below that Poor. The model's own overall impression is shown on the
   benchmark tab for comparison but never decides the grade, so the same pillar scores always
   give the same grade.
7. **Camera gate.** A serious camera failure (for example the tutor's face hidden for most of
   the class) makes the class a *quality issue*, graded Poor whatever the teaching, as the
   academy does. Smaller problems are reported but not graded: meeting the camera standard is
   expected, not a sign of good teaching.

Confidence drops when the transcript is mostly non-English or garbled, when no Zoom transcript
was available, or when the score is close to a grade boundary.

## Privacy

- **Video frames never leave the server.** Face and lighting checks run locally; no image is
  sent to any API.
- **Text sent to Anthropic** is the pseudonymised transcript only. **Audio is sent to Groq**
  only when no `.vtt` is provided, for speech-to-text.
- **Name removal is best-effort.** It catches Zoom display names, names after greetings and
  in direct address, and rare capitalised words. A common name misheard by the speech
  recogniser can slip through.
- **Uploads are deleted** after grading. The finished report (pseudonymised) is cached on the
  server by file hash so the same recording is not processed and billed twice.
- The app is behind a login. The confidential sample recordings are not in the repository;
  only batch IDs, labels and scores are published.

## Limitations

- Tuned and tested on 10 labelled recordings: a sanity check, not a validation.
- A missing branded background is not detected yet (one quality-issue class is missed for this reason).
- The judge varies by roughly ±0.15 points between runs on the same class.
- Without a `.vtt`, all students are merged into one speaker, so participation spread is approximate.
- Non-English or mixed-language classes get low confidence.
