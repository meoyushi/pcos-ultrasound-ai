import { useEffect } from "react";

/** Apply neo landing theme to document (dark bg, full-height root). Cleans up on unmount. */
export function useNeoPage() {
  useEffect(() => {
    document.documentElement.classList.add("neo-page-active");
    return () => {
      document.documentElement.classList.remove("neo-page-active");
    };
  }, []);
}
