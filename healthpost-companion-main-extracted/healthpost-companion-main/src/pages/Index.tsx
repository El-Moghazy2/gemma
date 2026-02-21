import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ClinicalWorkspace from "@/components/ClinicalWorkspace";
import AboutTab from "@/components/AboutTab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Stethoscope, Info } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10">
        <Tabs defaultValue="workspace" className="w-full">
          <TabsList className="w-full md:w-auto mb-8 p-1 bg-muted/60 backdrop-blur-sm">
            <TabsTrigger value="workspace" className="gap-2 data-[state=active]:shadow-sm">
              <Stethoscope className="w-4 h-4" />
              Clinical Workspace
            </TabsTrigger>
            <TabsTrigger value="about" className="gap-2 data-[state=active]:shadow-sm">
              <Info className="w-4 h-4" />
              Architecture & About
            </TabsTrigger>
          </TabsList>
          <TabsContent value="workspace" className="animate-fade-up">
            <ClinicalWorkspace />
          </TabsContent>
          <TabsContent value="about" className="animate-fade-up">
            <AboutTab />
          </TabsContent>
        </Tabs>
      </main>
      <Footer />
    </div>
  );
};

export default Index;