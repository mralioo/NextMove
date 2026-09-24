import { useState, useCallback, useRef } from 'react'

export function useSpeech() {
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const recognitionRef = useRef(null)
  
  // TTS: speak text
  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return
    window.speechSynthesis.cancel()
    // Strip markdown and latency badges
    const clean = text.replace(/\[Agent.*?\]/g, '').replace(/[*_#`]/g, '').trim()
    const utt = new SpeechSynthesisUtterance(clean)
    utt.rate = 0.92
    utt.pitch = 1.0
    utt.lang = 'en-GB'
    
    // Try to find a premium voice
    const voices = window.speechSynthesis.getVoices()
    const preferred = voices.find(v => v.name.includes('Google UK English Female')) ||
                      voices.find(v => v.name.includes('Microsoft Zira')) ||
                      voices.find(v => v.lang === 'en-GB') ||
                      voices.find(v => v.lang.startsWith('en'))
    if (preferred) utt.voice = preferred
    
    utt.onstart  = () => setIsSpeaking(true)
    utt.onend    = () => setIsSpeaking(false)
    utt.onerror  = () => setIsSpeaking(false)
    window.speechSynthesis.speak(utt)
  }, [])
  
  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel()
    setIsSpeaking(false)
  }, [])
  
  // STT: listen for voice input
  const startListening = useCallback((onResult) => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) { alert('Speech recognition not supported in this browser.'); return }
    const rec = new SR()
    rec.continuous = false
    rec.interimResults = true
    rec.lang = 'en-US'
    
    rec.onresult = (e) => {
      const t = Array.from(e.results).map(r => r[0].transcript).join('')
      setTranscript(t)
      if (e.results[0].isFinal) onResult(t)
    }
    
    rec.onend  = () => setIsListening(false)
    rec.onerror = () => setIsListening(false)
    rec.start()
    recognitionRef.current = rec
    setIsListening(true)
    setTranscript('')
  }, [])
  
  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [])
  
  return { speak, stopSpeaking, isSpeaking, startListening, stopListening, isListening, transcript }
}
