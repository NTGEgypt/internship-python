import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { X } from "lucide-react";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onReject: (reason: string) => void;
  loading: boolean;
  error: string | null;
};

const RejectOcrResultDialog = ({
  open,
  onOpenChange,
  onReject,
  loading,
  error,
}: Props) => {
  const [rejectionReason, setRejectionReason] = useState("");

  const handleReject = () => {
    if (!rejectionReason.trim()) {
      return;
    }

    onReject(rejectionReason);
  };

  const handleClose = () => {
    if (loading) return;

    setRejectionReason("");
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent
        showCloseButton={false}
        className="max-w-md"
      >
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle className="text-lg font-semibold text-red-700">
            Reject OCR Result
          </DialogTitle>

          <X
            className="h-5 w-5 cursor-pointer text-gray-500 hover:text-gray-900"
            onClick={handleClose}
          />
        </DialogHeader>

        <div className="space-y-4">

          <p className="text-sm text-gray-500">
            Please provide a reason for rejecting this ID.
          </p>

          <textarea
            value={rejectionReason}
            onChange={(e) => setRejectionReason(e.target.value)}
            placeholder="Enter rejection reason..."
            rows={4}
            disabled={loading}
            className="w-full resize-none rounded-lg border border-gray-300 p-3 text-sm outline-none focus:border-red-400 focus:ring-2 focus:ring-red-100 disabled:bg-gray-100"
          />

          {error && (
            <p className="text-sm text-red-600">
              {error}
            </p>
          )}

          <div className="flex justify-end gap-3">

            <button
              type="button"
              onClick={handleClose}
              disabled={loading}
              className="cursor-pointer rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>

            <button
              type="button"
              onClick={handleReject}
              disabled={loading || !rejectionReason.trim()}
              className="cursor-pointer rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Rejecting..." : "Confirm Reject"}
            </button>

          </div>

        </div>
      </DialogContent>
    </Dialog>
  );
};

export default RejectOcrResultDialog;
