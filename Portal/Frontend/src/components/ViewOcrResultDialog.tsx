import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { X } from "lucide-react";

import type { OcrResult } from "../features/ocr-results/types/ocrResult";
import { useOcrResultImage } from "../features/ocr-results/hooks/useOcrResultImage";
import { useOcrReview } from "../features/ocr-results/hooks/useOcrReview";
import RejectOcrResultDialog from "@/components/RejectOcrResultDialog";
    
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
  const [rejectDialogOpen, setRejectDialogOpen] = useState(false);

  const {
    imageUrl,
    loading: imageLoading,
    error: imageError,
  } = useOcrResultImage(
    result?.id ?? null,
    open
  );

  const {
    review,
    loading: reviewLoading,
    error: reviewError,
  } = useOcrReview();

  if (!result) return null;

  const isPending =
    result.reviewStatus?.toUpperCase() === "PENDING";

  const handleApprove = async () => {
    const response = await review(
      result.id,
      "approve",
      ""
    );

    if (response) {
      alert("Record approved successfully.");

      onOpenChange(false);

      window.location.reload();
    }
  };

  const handleReject = async (reason: string) => {
    const response = await review(
      result.id,
      "reject",
      reason
    );

    if (response) {
      alert("Record rejected successfully.");

      setRejectDialogOpen(false);
      onOpenChange(false);

      window.location.reload();
    }
  };

  return (
    <>
      {/* =========================
          MAIN DIALOG
      ========================== */}
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

            {/* ID IMAGE */}
            <section>
              <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
                ID Image
              </h3>

              <div className="flex min-h-[200px] items-center justify-center rounded-lg border border-gray-200 bg-gray-50 p-5">

                {imageLoading && (
                  <div className="flex flex-col items-center gap-2">
                    <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />

                    <p className="text-sm text-gray-500">
                      Loading image...
                    </p>
                  </div>
                )}

                {!imageLoading && imageError && (
                  <p className="text-sm text-red-500">
                    Failed to load image.
                  </p>
                )}

                {!imageLoading &&
                  !imageError &&
                  imageUrl && (
                    <img
                      src={imageUrl}
                      alt="National ID"
                      className="max-h-[400px] max-w-full rounded-md object-contain"
                    />
                  )}

                {!imageLoading &&
                  !imageError &&
                  !imageUrl && (
                    <p className="text-sm text-gray-400">
                      No image available.
                    </p>
                  )}
              </div>
            </section>

            {/* IDENTITY */}
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

            {/* ADDRESS */}
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

            {/* OCR & REVIEW */}
            <section>
              <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-blue-700">
                OCR & Review
              </h3>

              <div className="grid grid-cols-1 gap-5 rounded-lg border border-gray-200 bg-white p-5 md:grid-cols-2">

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

            {/* DECISION NOTE */}
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

            {/* FILE INFORMATION */}
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

            {/* APPROVE / REJECT */}
            {isPending && (
              <section className="border-t border-gray-200 pt-5">

                {reviewError && (
                  <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3">
                    <p className="text-sm text-red-600">
                      {reviewError}
                    </p>
                  </div>
                )}

                <div className="flex justify-end gap-3">

                  <button
                    type="button"
                    onClick={() => setRejectDialogOpen(true)}
                    disabled={reviewLoading}
                    className="cursor-pointer rounded-lg border border-red-200 bg-red-50 px-5 py-2 text-sm font-semibold text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Reject
                  </button>

                  <button
                    type="button"
                    onClick={handleApprove}
                    disabled={reviewLoading}
                    className="cursor-pointer rounded-lg bg-green-600 px-5 py-2 text-sm font-semibold text-white transition hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {reviewLoading
                      ? "Processing..."
                      : "Approve"}
                  </button>

                </div>
              </section>
            )}

          </div>
        </DialogContent>
      </Dialog>

      {/* REJECTION DIALOG */}
      <RejectOcrResultDialog
        open={rejectDialogOpen}
        onOpenChange={setRejectDialogOpen}
        onReject={handleReject}
        loading={reviewLoading}
        error={reviewError}
      />
    </>
  );
};

export default ViewOcrResultDialog;