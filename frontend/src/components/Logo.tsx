import { cn } from '@/utils/cn'

interface LogoProps {
  variant?: 'full' | 'mark'
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export default function Logo({ variant = 'full', size = 'md', className }: LogoProps) {
  const sizes = {
    sm: { icon: 22, text: 'text-sm', gap: 'gap-1.5' },
    md: { icon: 28, text: 'text-base', gap: 'gap-2' },
    lg: { icon: 34, text: 'text-xl', gap: 'gap-2.5' },
  }

  const s = sizes[size]

  return (
    <div className={cn('flex items-center', s.gap, className)}>
      {/* Icon mark */}
      <svg
        width={s.icon}
        height={s.icon}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Outer hex ring */}
        <path
          d="M16 2L28 9V23L16 30L4 23V9L16 2Z"
          stroke="#00A5D9"
          strokeWidth="1.5"
          strokeLinejoin="round"
          fill="none"
          opacity="0.6"
        />
        {/* Inner lightning / neural bolt */}
        <path
          d="M18 7L11 16H16L14 25L21 16H16L18 7Z"
          fill="url(#logo-grad)"
          strokeLinejoin="round"
        />
        <defs>
          <linearGradient id="logo-grad" x1="11" y1="7" x2="21" y2="25" gradientUnits="userSpaceOnUse">
            <stop stopColor="#33B8E3" />
            <stop offset="1" stopColor="#0070AD" />
          </linearGradient>
        </defs>
      </svg>

      {variant === 'full' && (
        <span className={cn('font-display font-bold tracking-tight', s.text)}>
          <span className="text-white">Edge</span>
          <span
            style={{
              background: 'linear-gradient(90deg, #00A5D9, #33B8E3, #00A5D9)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              fontWeight: 700,
            }}
          >
            Craft
          </span>
        </span>
      )}
    </div>
  )
}
