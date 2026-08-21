import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { X } from "lucide-react";
import type { OcrResult } from "../features/ocr-results/types/ocrResult";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  result: OcrResult | null;
};

function getStatusClasses(status: string) {
  switch (status?.toUpperCase()) {
    case "ACCEPTED":
      return "bg-green-100 text-green-700 border border-green-200";

    case "REJECTED":
      return "bg-red-100 text-red-700 border border-red-200";

    case "PENDING":
      return "bg-yellow-100 text-yellow-700 border border-yellow-200";

    default:
      return "bg-gray-100 text-gray-700 border border-gray-200";
  }
}

function getQualityClasses(quality: string) {
  switch (quality?.toUpperCase()) {
    case "GOOD":
      return "bg-green-100 text-green-700 border border-green-200";

    case "BORDERLINE":
      return "bg-yellow-100 text-yellow-700 border border-yellow-200";

    case "BAD":
      return "bg-red-100 text-red-700 border border-red-200";

    default:
      return "bg-gray-100 text-gray-700 border border-gray-200";
  }
}

function formatDate(value: string | null) {
  if (!value) return "N/A";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString();
}

function formatDateTime(value: string | null) {
  if (!value) return "N/A";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string | number | null | undefined;
}) {
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-gray-500">
        {label}
      </p>

      <p className="text-sm font-medium text-gray-900 break-words">
        {value || "N/A"}
      </p>
    </div>
  );
}

const ViewOcrResultDialog = ({
  open,
  onOpenChange,
  result,
}: Props) => {
  if (!result) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        className="max-w-3xl max-h-[90vh] overflow-y-auto scrollbar-hide"
      >
        {/* Header */}
        <DialogHeader className="flex flex-row items-center justify-between">
          <div>
            <DialogTitle className="text-xl font-semibold text-blue-900">
              OCR Result Details
            </DialogTitle>

            <p className="mt-1 text-sm text-gray-500">
              Extracted information from the national ID
            </p>
          </div>

          <X
            className="h-5 w-5 cursor-pointer text-gray-500 transition hover:text-gray-900"
            onClick={() => onOpenChange(false)}
          />
        </DialogHeader>

        <hr className="my-2 border-blue-100" />

        <div className="space-y-6">

          {/* Identity */}
          <section>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
              Identity Information
            </h3>

            <div className="grid grid-cols-1 gap-5 rounded-lg border border-blue-100 bg-blue-50/40 p-5 md:grid-cols-2">
              <Detail
                label="Full Name"
                value={result.fullName}
              />

              <Detail
                label="National ID"
                value={result.nationalId}
              />

              <Detail
                label="First Name"
                value={result.firstName}
              />

              <Detail
                label="Last Name"
                value={result.lastName}
              />

              <Detail
                label="Birth Date"
                value={formatDate(result.birthDate)}
              />

              <Detail
                label="Gender"
                value={result.gender}
              />

              <Detail
                label="Governorate"
                value={result.governorate}
              />

              <Detail
                label="Serial Number"
                value={result.serialNumber}
              />
            </div>
          </section>

          {/* Address */}
          <section>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
              Address
            </h3>

            <div className="rounded-lg border border-gray-200 bg-gray-50 p-5">
              <Detail
                label="Address"
                value={result.address}
              />
            </div>
          </section>

          {/* OCR & Review */}
          <section>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
              OCR & Review
            </h3>

            <div className="grid grid-cols-1 gap-5 rounded-lg border border-gray-200 bg-white p-5 md:grid-cols-2">

              {/* Quality */}
              <div>
                <p className="mb-1 text-sm font-medium text-gray-500">
                  Capture Quality
                </p>

                {result.captureQuality ? (
                  <span
                    className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${getQualityClasses(
                      result.captureQuality
                    )}`}
                  >
                    {result.captureQuality}
                  </span>
                ) : (
                  <span className="text-sm text-gray-400">
                    N/A
                  </span>
                )}
              </div>

              {/* Status */}
              <div>
                <p className="mb-1 text-sm font-medium text-gray-500">
                  Review Status
                </p>

                <span
                  className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${getStatusClasses(
                    result.reviewStatus
                  )}`}
                >
                  {result.reviewStatus}
                </span>
              </div>

              <Detail
                label="OCR Model Version"
                value={result.ocrModelVersion}
              />

              <Detail
                label="Reviewed By"
                value={result.reviewedBy}
              />

              <Detail
                label="Reviewed At"
                value={formatDateTime(result.reviewedAt)}
              />

              <Detail
                label="Created At"
                value={formatDateTime(result.createdAt)}
              />
            </div>
          </section>

          {/* Decision Note */}
          {result.decisionNote && (
            <section>
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-blue-700">
                Decision Note
              </h3>

              <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
                <p className="text-sm text-gray-700">
                  {result.decisionNote}
                </p>
              </div>
            </section>
          )}

          {/* File Information */}
          <section>
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
              File Information
            </h3>

            <div className="grid grid-cols-1 gap-5 rounded-lg border border-gray-200 bg-gray-50 p-5 md:grid-cols-2">
              <Detail
                label="Original Filename"
                value={result.originalFilename}
              />

              <Detail
                label="Image Type"
                value={result.imageMimeType}
              />

              <div className="md:col-span-2">
                <Detail
                  label="Image SHA-256"
                  value={result.imageSha256}
                />
              </div>
            </div>
          </section>

        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ViewOcrResultDialog;