# WALLX debug: synthesis (Claude, 2026-11-21 18:15)
Reviews: 2026-11-21_wallx_debug_fable.md, 2026-11-21_wallx_debug_astra.md. Packet: docs/reviews/wallx_debug_packet.md.
- session_id hypothesis (Fable #1) TESTED and rejected: identical requests spread 0.05-0.11 m with or without a fixed session_id,
  history on or off (sampling noise, not shared hidden state).
- Agreed causes, in order: (1) shoe-box bins look like graspable boxes -> use low-sided trays (shoe-box lids); (2) reset_history on
  every sub-goal / retry wipes context -> relay --no-reset-on-subgoal; (3) reply edits (--open-wide) diverge from what the model
  expects -> off for baselines (keep the idle-arm lock: 5/6 with it); (4) organiser-style wording "put the box into the {mat} bin";
  (5) stop at verified placement (done: success -> home -> hold); (6) no reorientation sub-goals (outside task B pick/place).
- Next test: recipe a4 = a3 with organiser wording, --no-reset-on-subgoal, no open-wide; then the same with lids as bins.
