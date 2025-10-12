import { IntellisenseItem } from '../vimEditorTypes';
import './intellisense-menu.css';

export class IntellisenseMenu {
  private container: HTMLDivElement | null = null;
  private items: IntellisenseItem[] = [];
  private selectedIndex: number = 0;
  private keydownHandler: ((e: KeyboardEvent) => void) | null = null;
  
  show(items: IntellisenseItem[], x: number, y: number, parentElement: HTMLElement): void {
    this.hide();
    
    this.items = items;
    this.selectedIndex = 0;
    
    this.container = document.createElement('div');
    this.container.className = 'intellisense-menu';
    this.container.style.position = 'absolute';
    this.container.style.left = `${x}px`;
    this.container.style.top = `${y + 20}px`;
    
    items.forEach((item, index) => {
      const itemElement = document.createElement('div');
      itemElement.className = 'intellisense-item';
      if (index === this.selectedIndex) {
        itemElement.classList.add('selected');
      }
      
      const textElement = document.createElement('div');
      textElement.className = 'intellisense-text';
      textElement.textContent = item.text;
      itemElement.appendChild(textElement);
      
      if (item.description) {
        const descElement = document.createElement('div');
        descElement.className = 'intellisense-description';
        descElement.textContent = item.description;
        itemElement.appendChild(descElement);
      }
      
      itemElement.addEventListener('click', () => {
        this.selectItem(index);
      });
      
      itemElement.addEventListener('mouseenter', () => {
        this.setSelectedIndex(index);
      });
      
      this.container!.appendChild(itemElement);
    });
    
    parentElement.appendChild(this.container);
    
    this.keydownHandler = this.handleKeyDown.bind(this);
    window.addEventListener('keydown', this.keydownHandler);
  }
  
  hide(): void {
    if (this.container) {
      this.container.remove();
      this.container = null;
    }
    
    if (this.keydownHandler) {
      window.removeEventListener('keydown', this.keydownHandler);
      this.keydownHandler = null;
    }
  }
  
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.container) return;
    
    switch (event.key) {
      case 'Escape':
        event.preventDefault();
        event.stopPropagation();
        this.hide();
        break;
        
      case 'ArrowUp':
        event.preventDefault();
        event.stopPropagation();
        this.moveSelection(-1);
        break;
        
      case 'ArrowDown':
        event.preventDefault();
        event.stopPropagation();
        this.moveSelection(1);
        break;
        
      case 'Enter':
        event.preventDefault();
        event.stopPropagation();
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
      if (index === this.selectedIndex) {
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
}

