import { useState, useRef } from 'react'
import { Info } from 'lucide-react'
import { cn } from '@/utils/cn'

interface InfoTooltipProps {
  text: string
  position?: 'top' | 'right' | 'bottom' | 'left'
  size?: number
  className?: string
}

export default function InfoTooltip({
  text, position = 'top', size = 14, className,
}: InfoTooltipProps) {
  const [visible, setVisible] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const posClass = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    right:  'left-full top-1/2 -translate-y-1/2 ml-2',
    left:   'right-full top-1/2 -translate-y-1/2 mr-2',
  }[position]

  return (
    <div
      ref={ref}
      className={cn('relative inline-flex items-center', className)}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <Info size={size} className="text-slate-500 hover:text-slate-400 cursor-help transition-colors" />
      {visible && (
        <div className={cn(
          'absolute z-50 w-56 px-3 py-2 text-xs text-slate-300 leading-relaxed',
          'bg-slate-800/95 backdrop-blur-md border border-white/10 rounded-lg shadow-xl',
          posClass,
        )}>
          {text}
        </div>
      )}
    </div>
  )
}
