import React, { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface PdfUploadProps {
  onFileUpload: (file: File) => void;
  uploadedFile: File | null;
}

export const PdfUpload: React.FC<PdfUploadProps> = ({ onFileUpload, uploadedFile }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(file => file.type === 'application/pdf');

    if (pdfFile) {
      onFileUpload(pdfFile);
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF file only.",
        variant: "destructive",
      });
    }
  }, [onFileUpload, toast]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === 'application/pdf') {
        onFileUpload(file);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a PDF file only.",
          variant: "destructive",
        });
      }
    }
  }, [onFileUpload, toast]);

  return (
    <Card className="p-8 border-2 border-dashed transition-all duration-300 hover:shadow-soft">
      <div
        className={`text-center transition-all duration-300 ${
          isDragOver ? 'scale-105' : ''
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {uploadedFile ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center w-16 h-16 mx-auto bg-gradient-primary rounded-full">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">{uploadedFile.name}</h3>
              <p className="text-muted-foreground">
                {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => document.getElementById('pdf-upload')?.click()}
            >
              Change File
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center w-16 h-16 mx-auto bg-gradient-soft rounded-full">
              <Upload className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Upload PDF Document</h3>
              <p className="text-muted-foreground">
                Drag and drop your PDF here, or click to browse
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => document.getElementById('pdf-upload')?.click()}
            >
              Choose File
            </Button>
            <div className="flex items-center justify-center text-sm text-muted-foreground">
              <AlertCircle className="w-4 h-4 mr-2" />
              PDF files only, max 10MB
            </div>
          </div>
        )}
        
        <input
          id="pdf-upload"
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleFileSelect}
        />
      </div>
    </Card>
  );
};