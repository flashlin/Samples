/// Base custom exception for the application.
sealed class AppException implements Exception {
  const AppException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() => '$runtimeType: $message${cause != null ? ' ($cause)' : ''}';
}

/// Thrown when fetching or resolving YouTube stream data fails.
final class RemoteSourceException extends AppException {
  const RemoteSourceException(super.message, {super.cause});
}

/// Thrown when local SQLite storage operations fail.
final class LocalStorageException extends AppException {
  const LocalStorageException(super.message, {super.cause});
}

/// Thrown when background downloading or file removal fails.
final class FileDownloadException extends AppException {
  const FileDownloadException(super.message, {super.cause});
}
