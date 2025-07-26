import React from 'react';
import { GraduationCap, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="text-center space-y-6 mb-12">
      <div className="flex items-center justify-center space-x-3">
        <div className="p-3 bg-gradient-primary rounded-full shadow-primary">
          <GraduationCap className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-hero bg-clip-text text-transparent">
          MCQ Craft
        </h1>
        <Sparkles className="w-6 h-6 text-accent animate-pulse" />
      </div>
      
      <div className="max-w-2xl mx-auto space-y-3">
        <p className="text-xl text-muted-foreground leading-relaxed">
          Transform your PDF documents into engaging multiple-choice questions instantly. 
          Perfect for educators, trainers, and content creators.
        </p>
        <p className="text-sm text-muted-foreground">
          Simply upload your PDF, select the pages, and let AI generate high-quality MCQs for you.
        </p>
      </div>

      <div className="text-xs text-muted-foreground flex items-center justify-center space-x-1">
        <span>Created with</span>
        <span className="text-accent">♥</span>
        <span>by</span>
        <span className="font-semibold text-primary">Shelton</span>
      </div>
    </header>
  );
};