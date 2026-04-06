import type { ReactNode } from 'react'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { motion } from 'framer-motion'
import NavBar from '@/components/NavBar'

interface WizardShellProps {
  step: number
  title: string
  description: string
  children: ReactNode
  onBack?: () => void
  onNext: () => void
  nextLabel?: string
  nextDisabled?: boolean
  nextLoading?: boolean
  footer?: ReactNode
}

export default function WizardShell({
  step, title, description, children,
  onBack, onNext, nextLabel = 'Continue',
  nextDisabled, nextLoading, footer,
}: WizardShellProps) {
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {/* Step indicator */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
              Step {step} of 4
            </p>
            <h1 className="text-2xl font-bold text-slate-100 font-display mb-1">{title}</h1>
            <p className="text-slate-400 text-sm mb-8">{description}</p>
          </motion.div>

          {/* Content */}
          <motion.div
            initial={{ opacity: 1, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 }}
          >
            {children}
          </motion.div>

          {/* Footer nav */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/[0.06]">
            <div>
              {onBack && (
                <button
                  onClick={onBack}
                  className="btn-secondary flex items-center gap-2 text-sm py-2.5"
                >
                  <ArrowLeft size={15} />
                  Back
                </button>
              )}
            </div>

            <div className="flex items-center gap-3">
              {footer}
              <button
                onClick={onNext}
                disabled={nextDisabled || nextLoading}
                className="btn-primary flex items-center gap-2 text-sm py-2.5"
              >
                {nextLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    {nextLabel}
                    <ArrowRight size={15} />
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="h-8" />
        </div>
      </div>
    </div>
  )
}
