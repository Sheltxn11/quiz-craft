import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, Sparkles } from 'lucide-react';
import { Header } from '@/components/Header';
import { PdfUpload } from '@/components/PdfUpload';
import { McqSettings } from '@/components/McqSettings';
import { McqResults } from '@/components/McqResults';
import { useToast } from '@/hooks/use-toast';

interface McqQuestion {
  question: string;
  options: string[];
  answer: string;
  explanation: string;
}

const Index = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [pageRange, setPageRange] = useState({ from: 5, to: 14 });
  const [numQuestions, setNumQuestions] = useState(5);
  const [questions, setQuestions] = useState<McqQuestion[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isProcessingPdf, setIsProcessingPdf] = useState(false);
  const [pdfProcessed, setPdfProcessed] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = (file: File) => {
    setUploadedFile(file);
    setQuestions([]);
    setPdfProcessed(false);
    toast({
      title: "File uploaded successfully",
      description: `${file.name} is ready for processing.`,
    });
  };

  const processPdf = async () => {
    if (!uploadedFile) return false;

    setIsProcessingPdf(true);
    const formData = new FormData();
    formData.append('pdf', uploadedFile);
    formData.append('from', String(pageRange.from));
    formData.append('to', String(pageRange.to));

    try {
      const response = await fetch('http://localhost:5000/generate-mcq', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process PDF');
      }

      const data = await response.json();
      console.log('PDF processed:', data);

      toast({
        title: "PDF processed successfully",
        description: "Your PDF has been processed and is ready for MCQ generation.",
      });

      setPdfProcessed(true);
      return true;
    } catch (error) {
      console.error('PDF processing error:', error);
      toast({
        title: "Error processing PDF",
        description: (error as Error).message,
        variant: "destructive",
      });
      return false;
    } finally {
      setIsProcessingPdf(false);
    }
  };

  const generateMcqs = async () => {
    setIsGenerating(true);

    try {
      // Create search query based on content
      const searchQuery = `volume density measurement physics mathematics pages ${pageRange.from} to ${pageRange.to}`;

      const requestData = {
        query: searchQuery,
        num_questions: numQuestions,
        difficulty: "medium",
        bloom_level: "application",
        min_page: pageRange.from,
        max_page: pageRange.to
      };

      console.log('Generating MCQs with:', requestData);

      const response = await fetch('http://localhost:5000/generate-mcqs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate MCQs');
      }

      const data = await response.json();
      console.log('MCQ response:', data);

      if (data.mcqs && data.mcqs.length > 0) {
        setQuestions(data.mcqs);
        toast({
          title: "MCQs generated successfully!",
          description: `Generated ${data.mcqs.length} questions from your PDF.`,
        });
      } else {
        throw new Error('No MCQs were generated. Please try with different content or settings.');
      }
    } catch (error) {
      console.error('MCQ generation error:', error);
      toast({
        title: "Error generating MCQs",
        description: (error as Error).message,
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateQuestions = async () => {
    if (!uploadedFile) {
      toast({
        title: "No file uploaded",
        description: "Please upload a PDF file first.",
        variant: "destructive",
      });
      return;
    }

    // Step 1: Process PDF if not already processed
    if (!pdfProcessed) {
      const success = await processPdf();
      if (!success) return;
    }

    // Step 2: Generate MCQs
    await generateMcqs();
  };

  const handleGenerateNew = () => {
    setQuestions([]);
    setUploadedFile(null);
    setPdfProcessed(false);
  };

  return (
    <div className="min-h-screen bg-gradient-soft">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <Header />
        
        {questions.length === 0 ? (
          <div className="space-y-8">
            <PdfUpload 
              onFileUpload={handleFileUpload} 
              uploadedFile={uploadedFile} 
            />
            
            {uploadedFile && (
              <>
                <McqSettings
                  pageRange={pageRange}
                  numQuestions={numQuestions}
                  onPageRangeChange={setPageRange}
                  onNumQuestionsChange={setNumQuestions}
                />
                
                <div className="text-center">
                  <Button
                    onClick={handleGenerateQuestions}
                    disabled={isGenerating || isProcessingPdf}
                    size="lg"
                    variant="hero"
                    className="text-lg px-8 py-4 h-auto"
                  >
                    {isProcessingPdf ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                        Processing PDF...
                      </>
                    ) : isGenerating ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                        Generating MCQs...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-3" />
                        {pdfProcessed ? 'Generate MCQs' : 'Process PDF & Generate MCQs'}
                      </>
                    )}
                  </Button>
                  {(isGenerating || isProcessingPdf) && (
                    <p className="text-sm text-muted-foreground mt-3">
                      {isProcessingPdf
                        ? `Processing PDF pages ${pageRange.from}-${pageRange.to}...`
                        : `Generating ${numQuestions} MCQs using AI...`
                      }
                    </p>
                  )}
                  {pdfProcessed && !isGenerating && !isProcessingPdf && (
                    <p className="text-sm text-green-600 mt-3">
                      ✅ PDF processed successfully. Ready to generate MCQs!
                    </p>
                  )}
                </div>
              </>
            )}
          </div>
        ) : (
          <McqResults 
            questions={questions} 
            onGenerateNew={handleGenerateNew} 
          />
        )}
      </div>
    </div>
  );
};

export default Index;
