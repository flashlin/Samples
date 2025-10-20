using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MnlStatementForIomCustomerRefEntityConfiguration : IEntityTypeConfiguration<MnlStatementForIomCustomerRefEntity>
    {
        public void Configure(EntityTypeBuilder<MnlStatementForIomCustomerRefEntity> builder)
        {
            builder.ToTable("MnlStatementForIomCustomerRef");


            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ReferenceNumber)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

        }
    }
}
