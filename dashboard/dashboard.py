import socket
import json
import sys
import math
import pygame
import tiles

# Configurations
ASSUMED_ALT_M = 150
HFOV_DEG = 110.0
VFOV_DEG = 82.0
COVERAGE_W_M = 2 * ASSUMED_ALT_M * math.tan(math.radians(HFOV_DEG / 2))
COVERAGE_H_M = 2 * ASSUMED_ALT_M * math.tan(math.radians(VFOV_DEG / 2))
PREDICT_HORIZON_S = 2.0
TRACK_TIMEOUT_S = 1.5
WIN_W, WIN_H = 1500, 950
PANEL_W = 360
MAP_W = WIN_W - PANEL_W
MAP_H = WIN_H
MAP_MARGIN = 90
C_BG = (14, 16, 22)
C_PANEL = (22, 26, 34)
C_GRID = (36, 42, 54)
C_FOOTPRINT = (44, 52, 68)
C_STATION = (90, 200, 160)
C_DRONE = (230, 70, 70)
C_OBJECT = (150, 158, 170)
C_TEXT = (210, 216, 226)
C_DIM = (130, 138, 150)
C_PATH = (120, 140, 200)
C_SELECT = (250, 210, 90)
DOT_R = 8

# Outbound-only line reader
class Stream:
    def __init__(self, addr):
        self.addr = addr
        self.sock = None
        self.buf = b""
        self.header = None
    def connect(self):
        s = socket.create_connection(self.addr, timeout=5)
        s.setblocking(False)
        self.sock = s
        self.buf = b""
        self.header = None
    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
    def poll_lines(self):
        if not self.sock:
            return []
        try:
            chunk = self.sock.recv(4096)
            if chunk == b"":
                raise ConnectionResetError("stream closed by Jetson")
            self.buf += chunk
        except BlockingIOError:
            pass
        lines = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if line:
                lines.append(line)
        return lines

# Coverage rectangle in screen pixels
def footprint_rect(backdrop):
    avail_w, avail_h = MAP_W - 2 * MAP_MARGIN, MAP_H - 2 * MAP_MARGIN
    if backdrop is not None:
        src_w, src_h = backdrop.get_width(), backdrop.get_height()
    else:
        src_w, src_h = COVERAGE_W_M, COVERAGE_H_M
    scale = min(avail_w / src_w, avail_h / src_h)
    rw, rh = src_w * scale, src_h * scale
    rx = (MAP_W - rw) / 2
    ry = (MAP_H - rh) / 2
    return rx, ry, rw, rh

# Map an object's frame-pixel center onto the backdrop
def frame_to_screen(px, py, w, h, rect):
    rx, ry, rw, rh = rect
    nx = (1.0 - py / h) if h else 0.5
    ny = (1.0 - px / w) if w else 0.5
    sx = rx + nx * rw
    sy = ry + ny * rh
    return sx, sy

# Render objects
def draw_backdrop(screen, font, rect, header, backdrop):
    rx, ry, rw, rh = rect
    if backdrop is not None:
        sw, sh = backdrop.get_width(), backdrop.get_height()
        factor = min(rw / sw, rh / sh)
        dw, dh = int(sw * factor), int (sh * factor)
        scaled = pygame.transform.smoothscale(backdrop, (dw, dh))
        bx = rx + (rw - dw) / 2
        by = ry + (rh - dh) / 2
        screen.blit(scaled, (int(bx), int(by)))
        rx, ry, rw, rh = bx, by, dw, dh
    else:
        for gx in range(0, MAP_W, 40):
            pygame.draw.line(screen, C_GRID, (gx, 0), (gx, MAP_H))
        for gy in range(0, MAP_H, 40):
            pygame.draw.line(screen, C_GRID, (0, gy), (MAP_W, gy))
    pygame.draw.rect(screen, C_FOOTPRINT, (rx, ry, rw, rh), width=2)
    lbl = font.render(f"coverage ~{COVERAGE_W_M:.0f} m x {COVERAGE_H_M:.0f} m @ {ASSUMED_ALT_M:.0f} m alt", True, C_DIM)
    screen.blit(lbl, (rx + 6, ry + 6))
    screen.blit(font.render("N", True, C_DIM), (rx + rw / 2 - 4, ry - 20))
    screen.blit(font.render("S", True, C_DIM), (rx + rw / 2 - 4, ry + rh + 6))

def draw_object(screen, font, obj, rect, header, selected):
    w = header['w']; h = header['h']
    sx, sy = frame_to_screen(obj['px'], obj['py'], w, h, rect)
    color = C_DRONE if obj['drone'] else C_OBJECT
    if selected:
        pygame.draw.circle(screen, C_SELECT, (int(sx), int(sy)), DOT_R + 5, width=2)
        fx = obj['px'] + obj['vx'] * PREDICT_HORIZON_S
        fy = obj['py'] + obj['vy'] * PREDICT_HORIZON_S
        ex, ey = frame_to_screen(fx, fy, w, h, rect)
        pygame.draw.line(screen, C_PATH, (sx, sy), (ex, ey), 2)
        pygame.draw.circle(screen, C_PATH, (int(ex), int(ey)), 3)
    pygame.draw.circle(screen, color, (int(sx), int(sy)), DOT_R)
    screen.blit(font.render(f"#{obj['id']}", True, C_TEXT), (sx + DOT_R + 2, sy - 8))
    obj['_screen'] = (sx, sy)
def draw_panel(screen, font, bigfont, objects, n, d, selected_id, connected):
    px0 = MAP_W
    pygame.draw.rect(screen, C_PANEL, (px0, 0, PANEL_W, WIN_H))
    y = 20
    screen.blit(bigfont.render("DETECTION FEED", True, C_TEXT), (px0 + 20, y)); y += 40
    status = "connected" if connected else "reconnecting..."
    scol = C_STATION if connected else C_SELECT
    screen.blit(font.render(status, True, scol), (px0 + 20, y)); y += 34
    screen.blit(font.render(f"objects: {n}", True, C_TEXT), (px0 + 20, y)); y += 24
    screen.blit(font.render(f"drones:  {d}", True, C_DRONE), (px0 + 20, y)); y += 36
    pygame.draw.line(screen, C_GRID, (px0 + 20, y), (px0 + PANEL_W - 20, y)); y += 16
    for obj in objects:
        sel = (obj['id'] == selected_id)
        col = C_SELECT if sel else (C_DRONE if obj['drone'] else C_TEXT)
        tag = "DRONE" if obj['drone'] else "object"
        screen.blit(font.render(f"#{obj['id']}  {tag}", True, col), (px0 + 20, y)); y += 22
        screen.blit(font.render(f"  at ({obj['px']:.0f},{obj['py']:.0f})", True, C_DIM), (px0 + 20, y)); y == 20
        speed = math.hypot(obj['vx'], obj['vy'])
        screen.blit(font.render(f"  v ({obj['vx']:.0f},{obj['vy']:.0f}) |v|={speed:.0f}px/s", True, C_DIM), (px0 + 20, y)); y += 26
        if y > WIN_H - 40:
            break
    hint = "click a dot to see its predicted path"
    screen.blit(font.render(hint, True, C_DIM), (px0 + 20, WIN_H - 30))

def main():
    arg = sys.argv[1] if len(sys.argv) == 2 else os.environ.get("DRONE_JETSON_ADDR")
    if not arg or ":" not in arg:
        print("usage: python3 dashboard.py <jetson_ip:port>", file=sys.stderr)
        print(" or: DRONE_JETSON_ADDR=<ip:port> python3 dashboard.py", file=sys.stderr)
        sys.exit(1)
    host, port = arg.rsplit(":", 1)
    addr = (host, int(port))

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Drone Detection Dashboard (local)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo, consolas, monospace", 15)
    bigfont = pygame.font.SysFont("menlo, consolas, monospace", 20, bold=True)

    stream = Stream(addr)
    header = None
    backdrop = None
    tracks = {}
    n_count = d_count = 0
    selected_id = None
    connected = False
    reconnect_at = 0.0

    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0
        if not connected and now >= reconnect_at:
            try:
                stream.connect()
                connected = True
                header = None
            except OSError:
                reconnect_at = now + 2.0
        if connected:
            try:
                for raw in stream.poll_lines():
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if "lat" in msg:
                        header = msg
                        tracks.clear()
                        if backdrop is None:
                            try:
                                backdrop = tiles.footprint_backdrop(header['lat'], header['lon'], COVERAGE_W_M, COVERAGE_H_M)
                            except Exception as e:
                                print(f"tiles: backdrop unavailable ({e}); using grid", file=sys.stderr)
                                backdrop = None
                        continue
                    if msg.get("error") == "busy":
                        print("refused: another dashboard is already connected", file=sys.stderr)
                        running = False
                        break
                    n_count = msg.get("n", 0)
                    d_count = msg.get("d", 0)
                    for (oid, drone, px, py, vx, vy) in msg.get("o", []):
                        tracks[oid] = {
                            "id" : oid, "drone" : bool(drone),
                            "px" : px, "py" : py, "vx" : vx, "vy" : vy,
                            "_last_seen" : now
                        }
            except (ConnectionResetError, OSError):
                connected = False
                stream.close()
                reconnect_at = now + 2.0
                header = None

        for oid in [k for k, v in tracks.items() if now - v['_last_seen'] > TRACK_TIMEOUT_S]:
            del tracks[oid]
        if not tracks:
            n_count = d_count = 0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                hit = None
                for obj in tracks.values():
                    sc = obj.get("_screen")
                    if sc and math.hypot(mx - sc[0], my - sc[1]) <= DOT_R + 6:
                        hit = obj['id']; break
                selected_id = hit

        screen.fill(C_BG)
        rect = footprint_rect(backdrop)
        draw_backdrop(screen, font, rect, header, backdrop)
        if header:
            ordered = sorted(tracks.values(), key=lambda o: o['id'])
            for obj in ordered:
                draw_object(screen, font, obj, rect, header, selected=(obj['id'] == selected_id))
            draw_panel(screen, font, bigfont, ordered, n_count, d_count, selected_id, connected)
        else:
            waiting = "waiting for telementry header..." if connected else "connecting to Jetson..."
            screen.blit(bigfont.render(waiting, True, C_DIM), (40, 40))
            draw_panel(screen, font, bigfont, [], 0, 0, None, connected)

        pygame.display.flip()
        clock.tick(60)

    stream.close()
    pygame.quit()

if __name__ == "__main__":
    main()
