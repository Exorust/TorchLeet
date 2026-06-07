"use client";

interface Props {
  company: string;
}

export default function CompanyTag({ company }: Props) {
  return (
    <span className="bg-lavender-100 text-lavender-700 text-xs rounded-full px-2 py-0.5 font-medium">
      {company}
    </span>
  );
}
