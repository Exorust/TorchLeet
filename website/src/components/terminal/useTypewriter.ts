"use client";

import { useState, useEffect } from "react";

export function useTypewriter(
  text: string,
  speed: number,
  startDelay?: number,
  onComplete?: () => void,
): { displayedText: string; isComplete: boolean } {
  const [displayedText, setDisplayedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    setDisplayedText("");
    setIsComplete(false);

    if (!text) {
      setIsComplete(true);
      return;
    }

    let index = 0;
    let timer: ReturnType<typeof setTimeout>;

    const tick = () => {
      if (index < text.length) {
        index++;
        setDisplayedText(text.slice(0, index));
        timer = setTimeout(tick, speed);
      } else {
        setIsComplete(true);
        onComplete?.();
      }
    };

    const delayTimer = setTimeout(tick, startDelay ?? 0);

    return () => {
      clearTimeout(delayTimer);
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, speed, startDelay]);

  return { displayedText, isComplete };
}
