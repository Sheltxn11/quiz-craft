import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CheckCircle, Copy, Download } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface McqQuestion {
  question: string;
  options: string[];
  answer: string;
  explanation: string;
}

interface McqResultsProps {
  questions: McqQuestion[];
  onGenerateNew: () => void;
}

export const McqResults: React.FC<McqResultsProps> = ({ questions, onGenerateNew }) => {
  const { toast } = useToast();

  const copyToClipboard = () => {
    const text = questions
      .map((q, index) => {
        const options = q.options.map((opt, i) => `${String.fromCharCode(65 + i)}. ${opt}`).join('\n   ');
        return `${index + 1}. ${q.question}\n   ${options}\n   Correct Answer: ${q.answer}\n   Explanation: ${q.explanation}\n`;
      })
      .join('\n');

    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard",
      description: "All questions have been copied to your clipboard.",
    });
  };

  const downloadQuestions = () => {
    const text = questions
      .map((q, index) => {
        const options = q.options.map((opt, i) => `${String.fromCharCode(65 + i)}. ${opt}`).join('\n   ');
        return `${index + 1}. ${q.question}\n   ${options}\n   Correct Answer: ${q.answer}\n   Explanation: ${q.explanation}\n`;
      })
      .join('\n');

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mcq-questions.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <CheckCircle className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold text-foreground">Generated Questions</h2>
          <Badge variant="secondary">{questions.length} Questions</Badge>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={copyToClipboard}>
            <Copy className="w-4 h-4 mr-2" />
            Copy All
          </Button>
          <Button variant="outline" size="sm" onClick={downloadQuestions}>
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
          <Button variant="hero" size="sm" onClick={onGenerateNew}>
            Generate New
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        {questions.map((question, index) => {
          // Convert answer letter to index (A=0, B=1, C=2, D=3)
          const correctAnswerIndex = question.answer.charCodeAt(0) - 65;

          return (
            <Card key={index} className="p-6 hover:shadow-soft transition-all duration-300">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <Badge variant="outline" className="mt-1">
                    {index + 1}
                  </Badge>
                  <div className="flex-1">
                    <h3 className="text-lg font-medium text-foreground leading-relaxed">
                      {question.question}
                    </h3>
                  </div>
                </div>

                <div className="ml-8 space-y-2">
                  {question.options.map((option, optionIndex) => (
                    <div
                      key={optionIndex}
                      className={`p-3 rounded-lg border transition-all duration-200 ${
                        optionIndex === correctAnswerIndex
                          ? 'bg-gradient-soft border-primary'
                          : 'bg-muted border-muted'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="font-medium text-primary">
                          {String.fromCharCode(65 + optionIndex)}.
                        </span>
                        <span className="text-foreground">{option}</span>
                        {optionIndex === correctAnswerIndex && (
                          <CheckCircle className="w-4 h-4 text-primary ml-auto" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {question.explanation && (
                  <div className="ml-8 p-3 bg-accent/20 rounded-lg border-l-4 border-accent">
                    <p className="text-sm text-muted-foreground">
                      <strong>Explanation:</strong> {question.explanation}
                    </p>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};