# Sim scene previews (scene v2, 2026-11-20)

Head camera with labels on top, both wrist cameras below. The arms hold still (no WALLX, nothing trained). Each frame is
t = 11 s, just after a parcel has been swiped in from the left. One QR label is on a random face each episode.

| Case | Object | Goes to |
|---|---|---|
| [18](case18.jpg) | Package 1, Amazon box 229 x 89 x 152 mm, QR 420/95112 | blue bin (lane A) |
| [19](case19.jpg) | Package 2, Amazon box 206 x 279 x 102 mm, QR 420/90210 | red bin (lane B), too wide for one gripper |
| [20](case20.jpg) | Package 3, padded mailer 216 x 254 x 13 mm, QR 420/10001 | black bin (unknown ZIP) |
| [21](case21.jpg) | 60 mm cube, no label | black bin |
| [22](case22.jpg) | Perfume box (not a parcel) | black bin |
| [24](case24.jpg) | Small box 76 mm, QR 420/94110 | blue bin |
| [25](case25.jpg) | Tall box 127 x 76 x 203 mm, QR 420/90015 | red bin |
| [26](case26.jpg) | White retail box 127 x 64 x 89 mm, QR 420/60601 | black bin (unknown ZIP) |
| [27](case27.jpg) | Large box 254 x 305 x 152 mm, QR 420/95014 | blue bin, too wide for one gripper |

Known issues in these frames, both fixed since: in 19 and 25 the parcel slides into the left gripper (the drop zone is now
clear of it), and the parcels still show the old plain QR label (the realistic shipping label from `docs/labels/` is next).
