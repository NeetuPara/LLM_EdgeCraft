import Dexie, { type Table } from 'dexie'

export interface Recipe {
  id: string
  name: string
  description?: string
  nodes: unknown[]
  edges: unknown[]
  createdAt: number
  updatedAt: number
}

class RecipesDatabase extends Dexie {
  recipes!: Table<Recipe>
  constructor() {
    super('unslothcraft-recipes')
    this.version(1).stores({ recipes: 'id, updatedAt' })
  }
}

export const recipesDb = new RecipesDatabase()

export async function createRecipe(name: string, nodes: unknown[] = [], edges: unknown[] = []): Promise<Recipe> {
  const recipe: Recipe = {
    id: crypto.randomUUID(), name, nodes, edges,
    createdAt: Date.now(), updatedAt: Date.now(),
  }
  await recipesDb.recipes.add(recipe)
  return recipe
}

export async function saveRecipe(id: string, updates: Partial<Recipe>) {
  await recipesDb.recipes.update(id, { ...updates, updatedAt: Date.now() })
}

export async function deleteRecipe(id: string) {
  await recipesDb.recipes.delete(id)
}
