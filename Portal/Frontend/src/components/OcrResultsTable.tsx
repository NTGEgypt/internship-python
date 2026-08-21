import { useMemo, useState } from "react";
import type { OcrResult } from "../features/ocr-results/types/ocrResult";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface OcrResultsTableProps {
  results: OcrResult[];
  onRowClick?: (result: OcrResult) => void;
}

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

export function OcrResultsTable({
  results,
  onRowClick,
}: OcrResultsTableProps) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("ALL");

  const filteredResults = useMemo(() => {
    const searchValue = search.toLowerCase().trim();

    return results.filter((result) => {
      const matchesSearch =
        !searchValue ||
        result.fullName?.toLowerCase().includes(searchValue) ||
        result.nationalId?.toLowerCase().includes(searchValue) ||
        result.governorate?.toLowerCase().includes(searchValue);

      const matchesStatus =
        statusFilter === "ALL" ||
        result.reviewStatus?.toUpperCase() === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [results, search, statusFilter]);

  const clearFilters = () => {
    setSearch("");
    setStatusFilter("ALL");
  };

  return (
    <div className="w-full px-6 py-6 md:px-10 lg:px-14">
      <div className="overflow-hidden rounded-xl border border-blue-200 bg-white shadow-lg shadow-blue-100/50">

        {/* Header */}
        <div className="border-b border-blue-200 bg-blue-50 px-6 py-4">
          <h2 className="text-lg font-semibold text-blue-900">
            OCR Results
          </h2>

          <p className="mt-1 text-sm text-blue-600">
            Review and manage extracted national ID records
          </p>
        </div>

        {/* Search & Filters */}
        <div className="flex flex-col gap-3 border-b border-blue-100 bg-white px-6 py-4 md:flex-row md:items-center md:justify-between">

          {/* Search */}
          <input
            type="text"
            placeholder="Search name, national ID, or governorate..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-10 w-full rounded-md border border-gray-300 px-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 md:max-w-md"
          />

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-2">

            {/* Clear */}
            {(search || statusFilter !== "ALL") && (
              <button
                type="button"
                onClick={clearFilters}
                className="h-10 rounded-md border border-gray-300 bg-white px-4 text-sm text-gray-600 transition hover:bg-gray-100"
              >
                Clear
              </button>
            )}

            {/* Status */}
            <div className="relative">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="h-10 appearance-none rounded-md border border-gray-300 bg-white py-0 pl-3 pr-9 text-sm text-gray-700 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              >
                <option value="ALL">All Statuses</option>
                <option value="ACCEPTED">Accepted</option>
                <option value="PENDING">Pending</option>
                <option value="REJECTED">Rejected</option>
              </select>

              <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                ▼
              </span>
            </div>

          </div>
        </div>

        {/* Result Count */}
        <div className="border-b border-blue-100 bg-gray-50 px-6 py-2">
          <span className="text-xs font-medium text-gray-500">
            Showing {filteredResults.length} of {results.length} results
          </span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <Table className="w-full">

            <TableHeader>
              <TableRow className="border-b border-blue-300 bg-blue-600 hover:bg-blue-600">

                <TableHead className="px-6 py-4 font-semibold text-white">
                  Name
                </TableHead>

                <TableHead className="px-6 py-4 font-semibold text-white">
                  National ID
                </TableHead>

                <TableHead className="px-6 py-4 font-semibold text-white">
                  Birth Date
                </TableHead>

                <TableHead className="px-6 py-4 font-semibold text-white">
                  Governorate
                </TableHead>

                <TableHead className="px-6 py-4 font-semibold text-white">
                  Status
                </TableHead>

                <TableHead className="px-6 py-4 font-semibold text-white">
                  Created At
                </TableHead>

              </TableRow>
            </TableHeader>

            <TableBody>
              {filteredResults.length > 0 ? (
                filteredResults.map((result, index) => (
                  <TableRow
                    key={result.id}
                    onClick={() => onRowClick?.(result)}
                    className={`
                      border-b border-blue-100
                      transition-colors
                      ${onRowClick ? "cursor-pointer" : ""}
                      ${index % 2 === 0 ? "bg-white" : "bg-blue-50/30"}
                      hover:bg-blue-100/50
                    `}
                  >

                    {/* Name */}
                    <TableCell className="px-6 py-4">
                      <span className="font-semibold text-gray-900">
                        {result.fullName ?? "-"}
                      </span>
                    </TableCell>

                    {/* National ID */}
                    <TableCell className="px-6 py-4">
                      <span className="font-mono text-sm text-gray-700">
                        {result.nationalId ?? "-"}
                      </span>
                    </TableCell>

                    {/* Birth Date */}
                    <TableCell className="px-6 py-4 text-gray-600">
                      {result.birthDate ?? "-"}
                    </TableCell>

                    {/* Governorate */}
                    <TableCell className="px-6 py-4 text-gray-600">
                      {result.governorate ?? "-"}
                    </TableCell>

                    {/* Status */}
                    <TableCell className="px-6 py-4">
                      <span
                        className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${getStatusClasses(
                          result.reviewStatus
                        )}`}
                      >
                        {result.reviewStatus}
                      </span>
                    </TableCell>

                    {/* Created At */}
                    <TableCell className="px-6 py-4 text-sm text-gray-500">
                      {result.createdAt}
                    </TableCell>

                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="h-32 text-center text-gray-500"
                  >
                    No OCR results found.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>

          </Table>
        </div>
      </div>
    </div>
  );
}