import React from "react";
import ChatCore from "../components/chat/ChatCore";

/** Chat Copilot: the same assistant as the floating one, full page (history on the left, live team progress / operations log on the right). */
export default function Chat() {
  return <div className="chat-page"><ChatCore variant="page" /></div>;
}
