import React from 'react';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { BookOpen, Hash } from 'lucide-react';

interface McqSettingsProps {
  pageRange: { from: number; to: number };
  numQuestions: number;
  onPageRangeChange: (range: { from: number; to: number }) => void;
  onNumQuestionsChange: (num: number) => void;
}

export const McqSettings: React.FC<McqSettingsProps> = ({
  pageRange,
  numQuestions,
  onPageRangeChange,
  onNumQuestionsChange,
}) => {
  return (
    <div className="space-y-6">
      <Card className="p-6 hover:shadow-soft transition-all duration-300">
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-primary" />
            <Label className="text-base font-semibold">Page Range</Label>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="page-from" className="text-sm">From Page</Label>
              <Input
                id="page-from"
                type="number"
                min="1"
                value={pageRange.from}
                onChange={(e) =>
                  onPageRangeChange({
                    ...pageRange,
                    from: parseInt(e.target.value) || 1,
                  })
                }
                className="focus:ring-primary"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="page-to" className="text-sm">To Page</Label>
              <Input
                id="page-to"
                type="number"
                min={pageRange.from}
                value={pageRange.to}
                onChange={(e) =>
                  onPageRangeChange({
                    ...pageRange,
                    to: parseInt(e.target.value) || 1,
                  })
                }
                className="focus:ring-primary"
              />
            </div>
          </div>
        </div>
      </Card>

      <Card className="p-6 hover:shadow-soft transition-all duration-300">
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Hash className="w-5 h-5 text-primary" />
            <Label className="text-base font-semibold">Number of Questions</Label>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Questions:</span>
              <span className="text-lg font-semibold text-primary">{numQuestions}</span>
            </div>
            <Slider
              value={[numQuestions]}
              onValueChange={(value) => onNumQuestionsChange(value[0])}
              max={50}
              min={1}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1</span>
              <span>50</span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};