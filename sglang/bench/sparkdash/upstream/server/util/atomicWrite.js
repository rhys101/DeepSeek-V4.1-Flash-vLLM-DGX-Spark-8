import fs from "fs";
import path from "path";

function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

/**
 * Atomic write that works even if an older root-owned 0600 file is in the way.
 *
 * Writes to a per-pid temp file, then renames onto the target. If rename fails
 * (e.g. cross-device or root-owned target), falls back to unlink + rename, then
 * to a direct overwrite as a last resort. Mode is best-effort applied via chmod
 * after write.
 *
 * @param {string} filePath Destination path.
 * @param {string} contents Text contents to write.
 * @param {number} [mode=0o600] File mode to apply (best-effort chmod).
 */
export function atomicWrite(filePath, contents, mode = 0o600) {
  ensureDir(filePath);
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, contents, { mode });
  try {
    fs.chmodSync(tmp, mode);
  } catch {
    /* best-effort */
  }
  try {
    fs.renameSync(tmp, filePath);
  } catch {
    // rename over root-owned file can fail — try unlink then rename / copy
    try {
      if (fs.existsSync(filePath)) {
        try {
          fs.chmodSync(filePath, 0o666);
        } catch {
          /* ignore */
        }
        fs.unlinkSync(filePath);
      }
      fs.renameSync(tmp, filePath);
    } catch (err2) {
      // Last resort: direct overwrite (may work when rename doesn't)
      try {
        fs.writeFileSync(filePath, contents, { mode });
        try {
          fs.unlinkSync(tmp);
        } catch {
          /* ignore */
        }
      } catch (err3) {
        try {
          fs.unlinkSync(tmp);
        } catch {
          /* ignore */
        }
        throw new Error(
          `Failed to write ${filePath}: ${err3.message}. ` +
            `If this is a root-owned file: sudo chown $(id -u):$(id -g) ${filePath}`
        );
      }
    }
  }
  try {
    fs.chmodSync(filePath, mode);
  } catch {
    /* best-effort */
  }
}

export default atomicWrite;