using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class RejectBetsLogEntityConfiguration : IEntityTypeConfiguration<RejectBetsLogEntity>
    {
        public void Configure(EntityTypeBuilder<RejectBetsLogEntity> builder)
        {
            builder.ToTable("RejectBetsLog");

            builder.HasKey(x => x.TransactionId);

            builder.Property(x => x.ID)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TransactionId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.RejectTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
