import cv2

def create_summary(path, fmap, idxs, scenes, out, fps, maxf):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))

    sel = fmap[idxs]
    written = 0

    if scenes:
        for sf in sel:
            for s, e in scenes:
                if s <= sf <= e:
                    cap.set(1, s)
                    for _ in range(s, e):
                        if written >= maxf:
                            break
                        r, f = cap.read()
                        if not r:
                            break
                        vw.write(f)
                        written += 1
                    break
            if written >= maxf:
                break

    # Fallback: if no matching scenes/frames were written, write from start.
    if written == 0:
        cap.set(1, 0)
        while written < maxf:
            r, f = cap.read()
            if not r:
                break
            vw.write(f)
            written += 1

    cap.release()
    vw.release()
    return written