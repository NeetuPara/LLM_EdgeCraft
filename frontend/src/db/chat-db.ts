import Dexie, { type Table } from 'dexie'

export interface Thread {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  modelName?: string
}

export interface Message {
  id: string
  threadId: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  modelName?: string
  streaming?: boolean
  imageDataUrl?: string   // base64 data URL for image attached by user (vision models)
}

class ChatDatabase extends Dexie {
  threads!: Table<Thread>
  messages!: Table<Message>

  constructor() {
    super('unslothcraft-chat')
    this.version(1).stores({
      threads: 'id, updatedAt',
      messages: 'id, threadId, timestamp',
    })
    // v2: add modelName index so RunDetailScreen can query threads by model efficiently
    this.version(2).stores({
      threads: 'id, updatedAt, modelName',
      messages: 'id, threadId, timestamp',
    })
  }
}

export const chatDb = new ChatDatabase()

export async function createThread(modelName?: string): Promise<Thread> {
  const thread: Thread = {
    id: crypto.randomUUID(),
    title: 'New Chat',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    modelName,
  }
  await chatDb.threads.add(thread)
  return thread
}

export async function updateThreadTitle(id: string, title: string) {
  await chatDb.threads.update(id, { title, updatedAt: Date.now() })
}

export async function deleteThread(id: string) {
  await chatDb.transaction('rw', chatDb.threads, chatDb.messages, async () => {
    await chatDb.messages.where('threadId').equals(id).delete()
    await chatDb.threads.delete(id)
  })
}

export async function addMessage(msg: Omit<Message, 'id' | 'timestamp'>): Promise<Message> {
  const message: Message = { ...msg, id: crypto.randomUUID(), timestamp: Date.now() }
  await chatDb.messages.add(message)
  return message
}

export async function updateMessage(id: string, updates: Partial<Message>) {
  await chatDb.messages.update(id, updates)
}

export async function getMessages(threadId: string): Promise<Message[]> {
  return chatDb.messages.where('threadId').equals(threadId).sortBy('timestamp')
}
