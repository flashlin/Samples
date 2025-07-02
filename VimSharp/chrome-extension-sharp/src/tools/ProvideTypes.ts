export enum ProvideKeys {
  LoadingState = 'loadingState',
} 

export interface LoadingState {
  isLoading: boolean;
}

//export const loadingState = inject(ProvideKeys.LoadingState) as LoadingState;