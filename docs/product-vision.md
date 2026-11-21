# BrickOrSwish: Product vision

What the app should become after the MVP, in the owner's words, with an assessment of each requirement:
can it be done, how, when, and what it changes in today's work. Milestones: **M1** Python prototype on recorded
video (now), **M2** live on the phone for one player, later milestones as listed per requirement.

---

## V1. Works on a court with other people, counts only my shots

> Later I would like it to be more resilient for everyday use. On a court there are often other people and it
> should still work: I go to the court with, say, 2 others and the app still counts my shots. Detecting me could
> work like this: the app asks me to stand in front of the camera and turn around, then it knows who the current
> player is and counts his shots.

**Feasible:** yes, with caveats (see Limits). **When:** after M2; it needs the live single-player app first.

### Three steps, each useful by itself

1. **Other people don't break it.** Two or three balls in the air near the rim, rebounds and passes must not
   create false shots or steal the track of a real one. The tracker follows a single ball today
   (`logic/tracker.py`); it becomes a multi-ball tracker where every ball has its own track and its own
   make/miss state machine. Needed even before identity: without it, friends' shots are counted as yours.
2. **Every shot gets a shooter.** At release, the ball's track is followed back to the person it left: the
   person box nearest the ball where its upward flight begins (hands/upper body). The detector already knows
   `person` (COCO). Today it only looks at the square around the hoop, so a second, cheap pass over the **whole
   frame** at ~2-5 fps finds the people; ball tracking near the rim stays as it is.
3. **Which shooter is me.** Enrollment as you describe: "stand in front of the camera and turn around" (~5 s).
   The app collects your appearance from all sides (front, back, sides) and builds a profile:
   - first version: **clothing colours** (shirt, shorts, shoes, sleeves) as colour histograms; no model needed,
     surprisingly strong for one session;
   - upgrade: a small **person re-identification** model (OSNet-class, a few MB) giving an appearance vector.
   Every shot's shooter is compared with the profile: **me / not me / not sure**.

### What the user sees
- Setup: tap the hoop (later automatic), then "Stand in front of the camera and turn around slowly".
- During play: only your shots count. Optionally everyone's: "Player 2 (red shorts): 5/9".
- After the session, "not sure" shots appear in a quick review with a thumbnail of the shooter: tap
  "me" / "not me". This review is the same idea as the verdict, so it also yields labels to improve the model.

### Limits
- **Similar clothing** (team kit, everyone in white) defeats colour matching; the re-ID model helps, and
  the review screen catches the rest.
- **Shooter out of frame at release**, or blocked by another player: the camera has to see where shots are
  taken from. Your current framing (phone low, looking up at the hoop, shooter in view) works.
- **Two shots at the same moment** from two people: fine once each ball has its own track (step 1).
- **Changing clothes** (jacket off) mid-session: redo the turn-around, or confirm in the review.

### Privacy (by design)
- On-device only; no images of other people are stored or uploaded. Profiles live only for the session.
- **No face recognition.** Matching uses body and clothing appearance, not faces: much less sensitive
  (faces are biometric data under GDPR) and it works from behind anyway.
- Only the person who enrols is identified; others are just "another player" unless the user names them.

### Compute (phone)
- Full-frame person pass at 2-5 fps: a fraction of what the hoop-crop ball detection already costs.
- Appearance check only at release moments (a few times per minute): negligible.
- Fine for iPhone 16; on an iPhone X the full-frame pass may need to drop to ~2 fps.

### What it changes now
- Keep `person` boxes in `detections.json` (the pipeline already keeps every class), and record some test
  sessions **with other players on court** for future fixtures.
- Write new shot-logic code so state lives **per ball track**, not globally, so going multi-ball is an
  extension, not a rewrite.
- `events.json` will gain an optional `shooter` field (schema version bump when it lands); nothing to do now.
