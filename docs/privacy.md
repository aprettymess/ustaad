# Privacy and safety model

- **Local only.** Video, frames, masks and model inference stay on the hub PC. No cloud calls.
- **Over-blur by default.** Default concept list (face, licence plate, house number, document, phone/computer screen,
  name badge, photograph) + concepts Glimmer finds in the scene. Masks are dilated and unioned across neighbouring
  sampled frames, so motion between samples is covered.
- **Audited.** Glimmer inspects the *redacted* output for anything still readable. Each leak gets an immediate box fix
  (± `fix_window_s`), plus a SAM re-track if it's a new concept. Fixed frames are re-checked. Unverified fixes and
  low-confidence detections go to a human review queue.
- **Context-aware.** Keep rules (e.g. a tyre size marking or an appliance model number) stay readable; personal text goes.
- **No biometrics.** No identification, recognition or matching of people. Click-to-unblur is manual and per-object.
- **Sharing boundary.** The raw master video is only viewable on the hub itself. Other devices get `redacted.mp4`
  and card images made from redacted frames. Learner evidence shows the master's *redacted* frames.
- **Learner attempts are private.** Feedback is for the learner only: no scores, no leaderboards.
- **Consent.** Everyone filmed must agree. The hero master's consent is recorded on camera.
- **Hazard refusal.** Mains electricity, gas, medical procedures and firearms are refused at the inventory stage.
  Procedures are labelled "community knowledge — not safety-certified".
