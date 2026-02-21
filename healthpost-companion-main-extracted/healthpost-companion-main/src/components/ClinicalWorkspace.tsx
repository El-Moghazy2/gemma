import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Mic, ImagePlus, Pill, Play, Loader2, FlaskConical } from "lucide-react";
import { DEMO_SCENARIOS, MOCK_REPORT } from "@/data/mockData";
import DiagnosticOutput from "./DiagnosticOutput";
import FollowUpChat from "./FollowUpChat";

const ClinicalWorkspace = () => {
  const [symptoms, setSymptoms] = useState("");
  const [age, setAge] = useState("");
  const [imageType, setImageType] = useState("Skin/Rash");
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);
  const [imageOpen, setImageOpen] = useState("");
  const [medsOpen, setMedsOpen] = useState("");

  const bothOpen = imageOpen === "image" && medsOpen === "meds";

  const handleDemoLoad = (value: string) => {
    const scenario = DEMO_SCENARIOS[value];
    if (scenario) {
      setSymptoms(scenario.symptoms);
      setAge(scenario.age);
    }
  };

  const handleRunWorkflow = () => {
    setLoading(true);
    setTimeout(() => {
      setReport(MOCK_REPORT);
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      {/* Demo Loader */}
      <div className="panel">
        <div className="flex items-center gap-2 mb-3">
          <FlaskConical className="w-4 h-4 text-primary" />
          <Label className="text-sm font-medium text-muted-foreground">
            Load a demo scenario (optional)
          </Label>
        </div>
        <Select onValueChange={handleDemoLoad}>
          <SelectTrigger className="w-full md:w-80 bg-background">
            <SelectValue placeholder="Select a demo scenario..." />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(DEMO_SCENARIOS).map((name) => (
              <SelectItem key={name} value={name}>{name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Patient Intake */}
      <div className="panel shadow-elevated">
        <h3 className="panel-header flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full gradient-primary inline-block shadow-sm" />
          Patient Information
        </h3>
        <div className="space-y-5">
          <div>
            <Label htmlFor="symptoms" className="mb-1.5 block text-sm font-medium">
              Symptoms
            </Label>
            <Textarea
              id="symptoms"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              placeholder="Describe the patient's symptoms in detail..."
              className="min-h-[120px] bg-background/50 focus:bg-background transition-colors"
              rows={4}
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <Label htmlFor="age" className="mb-1.5 block text-sm font-medium">Patient Age</Label>
              <Input
                id="age"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="e.g., adult, child 5 years"
                className="bg-background/50 focus:bg-background transition-colors"
              />
            </div>
            <div>
              <Label className="mb-1.5 block text-sm font-medium">Or record symptoms (MedASR)</Label>
              <button className="w-full flex items-center gap-3 p-3 rounded-xl border-2 border-dashed border-border bg-background/50 hover:border-primary/30 hover:bg-accent/50 transition-all cursor-pointer group">
                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <Mic className="w-4.5 h-4.5 text-primary" />
                </div>
                <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                  Click to record audio
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Attachments */}
      <div className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${bothOpen ? "items-stretch" : "items-start"}`}>
        <Accordion type="single" collapsible value={imageOpen} onValueChange={setImageOpen} className={bothOpen ? "h-full" : ""}>
          <AccordionItem value="image" className={`panel border-0 ${bothOpen ? "h-full" : ""}`}>
            <AccordionTrigger className="py-0 hover:no-underline">
              <span className="flex items-center gap-2.5 text-sm font-medium">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <ImagePlus className="w-4 h-4 text-primary" />
                </div>
                Add Medical Image (optional)
              </span>
            </AccordionTrigger>
            <AccordionContent className="pt-4 space-y-4">
              <div className="border-2 border-dashed border-border rounded-xl p-8 text-center bg-background/50 hover:border-primary/30 hover:bg-accent/30 transition-all cursor-pointer group">
                <ImagePlus className="w-8 h-8 text-muted-foreground mx-auto mb-2 group-hover:text-primary transition-colors" />
                <p className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                  Upload a medical photo
                </p>
              </div>
              <div>
                <Label className="mb-2 block text-sm font-medium">Image Type</Label>
                <RadioGroup value={imageType} onValueChange={setImageType} className="flex flex-wrap gap-3">
                  {["Skin/Rash", "Wound", "Eyes", "Other"].map((type) => (
                    <div key={type} className="flex items-center gap-1.5">
                      <RadioGroupItem value={type} id={type} />
                      <Label htmlFor={type} className="text-sm cursor-pointer">{type}</Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <Accordion type="single" collapsible value={medsOpen} onValueChange={setMedsOpen} className={bothOpen ? "h-full" : ""}>
          <AccordionItem value="meds" className={`panel border-0 ${bothOpen ? "h-full" : ""}`}>
            <AccordionTrigger className="py-0 hover:no-underline">
              <span className="flex items-center gap-2.5 text-sm font-medium">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Pill className="w-4 h-4 text-primary" />
                </div>
                Current Medications (optional)
              </span>
            </AccordionTrigger>
            <AccordionContent className="pt-4 space-y-4">
              <div className="border-2 border-dashed border-border rounded-xl p-8 text-center bg-background/50 hover:border-primary/30 hover:bg-accent/30 transition-all cursor-pointer group">
                <Pill className="w-8 h-8 text-muted-foreground mx-auto mb-2 group-hover:text-primary transition-colors" />
                <p className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                  Upload photo of medications
                </p>
              </div>
              <Textarea
                placeholder="Or type medication names, one per line"
                className="bg-background/50 focus:bg-background transition-colors"
                rows={3}
              />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      {/* Run Button */}
      <Button
        onClick={handleRunWorkflow}
        disabled={loading || (!symptoms && !age)}
        className="w-full h-14 text-base font-semibold gradient-primary border-0 shadow-lg hover:shadow-xl hover:scale-[1.01] transition-all duration-200"
        size="lg"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
            Running Complete Workflow...
          </>
        ) : (
          <>
            <Play className="w-5 h-5 mr-2" />
            Run Complete Workflow
          </>
        )}
      </Button>

      {/* Diagnostic Output */}
      {report && (
        <div className="animate-fade-up">
          <DiagnosticOutput report={report} />
        </div>
      )}

      {/* Follow-up Chat */}
      {report && (
        <div className="animate-fade-up" style={{ animationDelay: "0.1s" }}>
          <FollowUpChat />
        </div>
      )}
    </div>
  );
};

export default ClinicalWorkspace;