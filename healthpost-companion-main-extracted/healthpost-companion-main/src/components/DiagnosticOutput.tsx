import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText } from "lucide-react";

interface DiagnosticOutputProps {
  report: string;
}

const DiagnosticOutput = ({ report }: DiagnosticOutputProps) => (
  <div className="panel shadow-elevated">
    <h3 className="panel-header flex items-center gap-2">
      <span className="w-2.5 h-2.5 rounded-full bg-clinical-success inline-block shadow-sm" />
      <FileText className="w-5 h-5 text-clinical-success" />
      Diagnostic Report
    </h3>
    <div className="prose prose-sm max-w-none prose-headings:font-display prose-headings:text-foreground prose-p:text-foreground prose-td:text-foreground prose-th:text-foreground prose-th:bg-accent prose-th:px-3 prose-th:py-2 prose-td:px-3 prose-td:py-2 prose-table:border-border prose-blockquote:border-primary prose-blockquote:text-muted-foreground prose-table:rounded-lg prose-blockquote:bg-accent/30 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:rounded-lg">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
    </div>
  </div>
);

export default DiagnosticOutput;