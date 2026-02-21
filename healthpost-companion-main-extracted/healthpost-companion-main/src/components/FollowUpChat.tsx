import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Bot, User, MessageCircle } from "lucide-react";
import { CHAT_RESPONSES } from "@/data/mockData";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const FollowUpChat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input };
    const lower = input.toLowerCase();
    let response = CHAT_RESPONSES.default;
    if (lower.includes("pregnant")) response = CHAT_RESPONSES.pregnant;
    else if (lower.includes("dos") || lower.includes("dosage")) response = CHAT_RESPONSES.dosage;
    else if (lower.includes("refer")) response = CHAT_RESPONSES.referral;

    setMessages((prev) => [...prev, userMsg, { role: "assistant", content: response }]);
    setInput("");
  };

  return (
    <div className="panel shadow-elevated">
      <h3 className="panel-header flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full gradient-primary inline-block shadow-sm" />
        <MessageCircle className="w-5 h-5 text-primary" />
        Follow-up Questions
      </h3>
      <p className="text-sm text-muted-foreground mb-4">
        Ask questions about the diagnosis, dosage, referral criteria, etc.
      </p>

      <div className="bg-background/50 rounded-xl border border-border min-h-[200px] max-h-[300px] overflow-y-auto p-4 space-y-4 mb-4">
        {messages.length === 0 && (
          <div className="text-center py-10">
            <MessageCircle className="w-8 h-8 text-muted-foreground/30 mx-auto mb-2" />
            <p className="text-sm text-muted-foreground italic">
              No messages yet. Ask a follow-up question below.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}>
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-xl gradient-primary flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                <Bot className="w-4 h-4 text-white" />
              </div>
            )}
            <div
              className={`rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed ${
                msg.role === "user"
                  ? "gradient-primary text-white shadow-sm"
                  : "bg-accent/60 text-accent-foreground border border-border/50"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="whitespace-pre-wrap" dangerouslySetInnerHTML={{
                  __html: msg.content
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                    .replace(/\n/g, "<br/>")
                    .replace(/- /g, "• ")
                }} />
              ) : (
                msg.content
              )}
            </div>
            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-xl bg-secondary flex items-center justify-center shrink-0 mt-0.5 border border-border">
                <User className="w-4 h-4 text-secondary-foreground" />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="e.g., What if the patient is pregnant?"
          className="bg-background/50 focus:bg-background transition-colors rounded-xl"
        />
        <Button onClick={handleSend} size="icon" className="shrink-0 gradient-primary border-0 rounded-xl shadow-sm hover:shadow-md transition-shadow">
          <Send className="w-4 h-4 text-white" />
        </Button>
      </div>
    </div>
  );
};

export default FollowUpChat;