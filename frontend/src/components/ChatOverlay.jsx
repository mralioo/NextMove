import React, { useState } from "react";
import Toby, { loadPos } from "./Toby";
import ChatCore from "./chat/ChatCore";
import { useChat } from "../state/ChatContext";

/** Toby floats over every page; click = the conversation opens in the middle of the page (backdrop click or ✕ sends Toby back to his corner). */
export default function ChatOverlay() {
  const { open, setOpen, thinking, pending } = useChat();
  const [pos, setPosState] = useState(loadPos);
  const setPos = (p) => { setPosState(p); try { localStorage.setItem("toby.pos", JSON.stringify(p)); } catch { /* private mode */ } };
  return (
    <>
      <Toby open={open} onToggle={() => setOpen(!open)} badge={pending} thinking={thinking} hint={pending ? `${pending} to close` : "Ask me"} pos={pos} setPos={setPos} />
      {open && (
        <div className="chat-backdrop" onClick={() => setOpen(false)}>
          <div className="chat-frame" onClick={(e) => e.stopPropagation()}>
            <ChatCore variant="overlay" onClose={() => setOpen(false)} onDock={(c) => { setPos({ corner: c }); setOpen(false); }} />
          </div>
        </div>
      )}
    </>
  );
}
