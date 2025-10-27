import { IntellisenseItem } from '../vimEditorTypes';
import './intellisense-menu.css';

export class IntellisenseMenu {
  private container: HTMLDivElement | null = null;
  private items: IntellisenseItem[] = [];
  private selectedIndex: number = 0;
  private keydownHandler: ((e: KeyboardEvent) => void) | null = null;
  private x: number = 0;
  private y: number = 0;
  private parentElement: HTMLElement | null = null;
  private filterDisplay: HTMLDivElement | null = null;
  private currentFilterText: string = '';
  
  show(items: IntellisenseItem[], x: number, y: number, parentElement: HTMLElement): void {
    this.hide();
    
    this.items = items;
    this.selectedIndex = 0;
    this.x = x;
    this.y = y;
    this.parentElement = parentElement;
    
    this.container = document.createElement('div');
    this.container.className = 'intellisense-menu';
    const maxVisibleItems = 8;
    const itemHeight = 42;
    const filterDisplayHeight = 30;
    const maxHeight = Math.min(items.length, maxVisibleItems) * itemHeight + filterDisplayHeight;
    
    this.container.style.cssText = `
      position: absolute;
      left: ${x}px;
      top: ${y + 20}px;
      background: #3a3a3a;
      border: 2px solid #555;
      border-radius: 4px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      max-height: ${maxHeight}px;
      overflow-y: visible;
      z-index: 10000;
      min-width: 300px;
      font-family: 'Courier New', monospace;
    `;
    
    this.filterDisplay = document.createElement('div');
    this.filterDisplay.className = 'intellisense-filter-display';
    this.filterDisplay.style.cssText = `
      padding: 6px 14px;
      background: #2a2a2a;
      border-bottom: 1px solid #555;
      color: #ffd700;
      font-size: 13px;
      font-weight: bold;
      font-family: 'Courier New', monospace;
      border-radius: 2px 2px 0 0;
      position: sticky;
      top: 0;
      z-index: 1;
    `;
    this.updateFilterDisplay(this.currentFilterText);
    this.container.appendChild(this.filterDisplay);
    
    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'intellisense-items-container';
    itemsContainer.style.cssText = `
      max-height: ${Math.min(items.length, maxVisibleItems) * itemHeight}px;
      overflow-y: auto;
    `;
    this.container.appendChild(itemsContainer);
    
    items.forEach((item, index) => {
      const itemElement = document.createElement('div');
      itemElement.className = 'intellisense-item';
      const isSelected = index === this.selectedIndex;
      
      itemElement.style.cssText = `
        padding: 10px 14px;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        border-bottom: 1px solid #4a4a4a;
        background: ${isSelected ? '#1e3a5f' : '#3a3a3a'};
      `;
      
      if (isSelected) {
        itemElement.classList.add('selected');
      }
      
      const textElement = document.createElement('div');
      textElement.className = 'intellisense-text';
      textElement.textContent = item.text;
      textElement.style.cssText = `
        font-weight: bold;
        color: ${isSelected ? '#ffd700' : '#ffffff'};
        font-size: 14px;
      `;
      itemElement.appendChild(textElement);
      
      if (item.description) {
        const descElement = document.createElement('div');
        descElement.className = 'intellisense-description';
        descElement.textContent = item.description;
        descElement.style.cssText = `
          font-size: 12px;
          color: ${isSelected ? '#ffeb99' : '#cccccc'};
          margin-top: 2px;
        `;
        itemElement.appendChild(descElement);
      }
      
      itemElement.addEventListener('click', () => {
        this.selectItem(index);
      });
      
      itemElement.addEventListener('mouseenter', () => {
        this.setSelectedIndex(index);
      });
      
      itemElement.addEventListener('mouseover', () => {
        if (index !== this.selectedIndex) {
          itemElement.style.background = '#4a4a4a';
        }
      });
      
      itemElement.addEventListener('mouseout', () => {
        if (index !== this.selectedIndex) {
          itemElement.style.background = '#3a3a3a';
        }
      });
      
      itemsContainer.appendChild(itemElement);
    });
    
    parentElement.appendChild(this.container);
    
    this.keydownHandler = this.handleKeyDown.bind(this);
    window.addEventListener('keydown', this.keydownHandler, true);
  }
  
  hide(): void {
    if (this.container) {
      this.container.remove();
      this.container = null;
    }
    
    if (this.keydownHandler) {
      window.removeEventListener('keydown', this.keydownHandler, true);
      this.keydownHandler = null;
    }
    
    this.filterDisplay = null;
    this.currentFilterText = '';
  }
  
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.container) return;
    
    switch (event.key) {
      case 'Escape':
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        this.hide();
        break;
        
      case 'ArrowUp':
      case 'Up':
      case 'k':
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        this.moveSelection(-1);
        break;
        
      case 'ArrowDown':
      case 'Down':
      case 'j':
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        this.moveSelection(1);
        break;
        
      case 'Enter':
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        this.selectItem(this.selectedIndex);
        break;
    }
  }
  
  private moveSelection(delta: number): void {
    this.setSelectedIndex(this.selectedIndex + delta);
  }
  
  private setSelectedIndex(index: number): void {
    if (index < 0) index = 0;
    if (index >= this.items.length) index = this.items.length - 1;
    
    this.selectedIndex = index;
    this.updateSelection();
    this.scrollToSelected();
  }
  
  private updateSelection(): void {
    if (!this.container) return;
    
    const itemElements = this.container.querySelectorAll('.intellisense-item');
    itemElements.forEach((element, index) => {
      const isSelected = index === this.selectedIndex;
      const htmlElement = element as HTMLElement;
      
      htmlElement.style.background = isSelected ? '#1e3a5f' : '#3a3a3a';
      
      const textElement = htmlElement.querySelector('.intellisense-text') as HTMLElement;
      if (textElement) {
        textElement.style.color = isSelected ? '#ffd700' : '#ffffff';
      }
      
      const descElement = htmlElement.querySelector('.intellisense-description') as HTMLElement;
      if (descElement) {
        descElement.style.color = isSelected ? '#ffeb99' : '#cccccc';
      }
      
      if (isSelected) {
        element.classList.add('selected');
      } else {
        element.classList.remove('selected');
      }
    });
  }
  
  private scrollToSelected(): void {
    if (!this.container) return;
    
    const selectedElement = this.container.querySelector('.intellisense-item.selected');
    if (selectedElement) {
      selectedElement.scrollIntoView({ block: 'nearest' });
    }
  }
  
  private selectItem(index: number): void {
    if (index >= 0 && index < this.items.length) {
      const item = this.items[index];
      this.hide();
      item.action();
    }
  }
  
  updateItems(items: IntellisenseItem[], filterText?: string): void {
    this.items = items;
    this.selectedIndex = Math.min(this.selectedIndex, items.length - 1);
    if (this.selectedIndex < 0) this.selectedIndex = 0;
    
    if (filterText !== undefined) {
      this.currentFilterText = filterText;
      this.updateFilterDisplay(filterText);
    }
    
    if (this.container && this.parentElement) {
      this.renderItems();
    }
  }
  
  filterItems(filterText: string, originalItems: IntellisenseItem[]): IntellisenseItem[] {
    if (!filterText) {
      return originalItems;
    }
    
    const lowerFilter = filterText.toLowerCase();
    
    return originalItems.filter(item => {
      const lowerText = item.text.toLowerCase();
      let filterIndex = 0;
      
      for (let i = 0; i < lowerText.length && filterIndex < lowerFilter.length; i++) {
        if (lowerText[i] === lowerFilter[filterIndex]) {
          filterIndex++;
        }
      }
      
      return filterIndex === lowerFilter.length;
    });
  }
  
  private renderItems(): void {
    if (!this.container) return;
    
    const itemsContainer = this.container.querySelector('.intellisense-items-container');
    if (!itemsContainer) return;
    
    itemsContainer.innerHTML = '';
    
    if (this.items.length === 0) {
      const emptyElement = document.createElement('div');
      emptyElement.className = 'intellisense-empty';
      emptyElement.textContent = 'No matches found';
      emptyElement.style.cssText = `
        padding: 10px 14px;
        color: #999;
        font-style: italic;
        text-align: center;
      `;
      itemsContainer.appendChild(emptyElement);
      return;
    }
    
    this.items.forEach((item, index) => {
      const itemElement = document.createElement('div');
      itemElement.className = 'intellisense-item';
      const isSelected = index === this.selectedIndex;
      
      itemElement.style.cssText = `
        padding: 10px 14px;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        border-bottom: 1px solid #4a4a4a;
        background: ${isSelected ? '#1e3a5f' : '#3a3a3a'};
      `;
      
      if (isSelected) {
        itemElement.classList.add('selected');
      }
      
      const textElement = document.createElement('div');
      textElement.className = 'intellisense-text';
      textElement.textContent = item.text;
      textElement.style.cssText = `
        font-weight: bold;
        color: ${isSelected ? '#ffd700' : '#ffffff'};
        font-size: 14px;
      `;
      itemElement.appendChild(textElement);
      
      if (item.description) {
        const descElement = document.createElement('div');
        descElement.className = 'intellisense-description';
        descElement.textContent = item.description;
        descElement.style.cssText = `
          font-size: 12px;
          color: ${isSelected ? '#ffeb99' : '#cccccc'};
          margin-top: 2px;
        `;
        itemElement.appendChild(descElement);
      }
      
      itemElement.addEventListener('click', () => {
        this.selectItem(index);
      });
      
      itemElement.addEventListener('mouseenter', () => {
        this.setSelectedIndex(index);
      });
      
      itemElement.addEventListener('mouseover', () => {
        if (index !== this.selectedIndex) {
          itemElement.style.background = '#4a4a4a';
        }
      });
      
      itemElement.addEventListener('mouseout', () => {
        if (index !== this.selectedIndex) {
          itemElement.style.background = '#3a3a3a';
        }
      });
      
      itemsContainer.appendChild(itemElement);
    });
  }
  
  isVisible(): boolean {
    return this.container !== null;
  }
  
  updateFilterDisplay(filterText: string): void {
    if (!this.filterDisplay) return;
    
    const displayText = filterText || '';
    const cursor = '▌';
    this.filterDisplay.innerHTML = `<span style="color: #aaa;">Filter:</span> ${displayText}<span style="animation: blink 1s infinite;">${cursor}</span>`;
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
      }
    `;
    
    if (!document.querySelector('#intellisense-blink-style')) {
      style.id = 'intellisense-blink-style';
      document.head.appendChild(style);
    }
  }
}

